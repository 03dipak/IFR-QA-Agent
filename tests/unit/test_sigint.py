"""LLD-46: SIGINT handling and resumable bulk ingestion."""

from __future__ import annotations

import hashlib
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest

from src.ingestion.ledger import Ledger
from src.ingestion.models import CandidateState

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_pdf(text: str) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    for _ in range(3):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
        })
        stream = DecodedStreamObject()
        stream.set_data(
            f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


class _SlowServer(ThreadingHTTPServer):
    daemon_threads = True
    fast_pdf: bytes
    slow_pdf: bytes
    fast: bool
    hits: dict[str, int]
    lock: threading.Lock
    started: threading.Event
    release: threading.Event
    base_url: str


class _SlowHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        server = cast(_SlowServer, self.server)
        with server.lock:
            server.hits[self.path] = server.hits.get(self.path, 0) + 1
        body = server.fast_pdf if self.path == "/fast.pdf" else server.slow_pdf
        fast = self.path == "/fast.pdf" or server.fast
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            if fast:
                self.wfile.write(body)
                return
            self.wfile.write(body[:20])
            self.wfile.flush()
            server.started.set()
            server.release.wait(timeout=20)
            self.wfile.write(body[20:])
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args, **kwargs) -> None:
        return


@pytest.fixture()
def slow_server():
    server = _SlowServer(("127.0.0.1", 0), _SlowHandler)
    server.fast_pdf = _make_pdf(
        "Balance Sheet total assets 100 revenue 200 liabilities 50 cash flow 10 profit 5")
    server.slow_pdf = _make_pdf(
        "Statement of Profit and Loss revenue 300 total assets 400 cash flow 20 profit 5 tax 1 net 2")
    server.fast = False
    server.hits = {}
    server.lock = threading.Lock()
    server.started = threading.Event()
    server.release = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        address = cast("tuple[str, int]", server.server_address)
        server.base_url = f"http://{address[0]}:{address[1]}"
        yield server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_csv(path: Path, base_url: str) -> None:
    path.write_text(
        "company_id,document_type,fiscal_period,url\n"
        f"FAST,annual,FY2024,{base_url}/fast.pdf\n"
        f"SLOW,annual,FY2023,{base_url}/slow.pdf\n",
        encoding="utf-8",
    )


def _start_bulk(csv_path: Path, ledger_path: Path, raw_path: Path):
    return subprocess.Popen(
        [
            sys.executable, "-m", "src.ingestion.pipeline", "bulk",
            "--candidates", str(csv_path),
            "--ledger", str(ledger_path),
            "--storage-dir", str(raw_path),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(proc: subprocess.Popen, timeout: float = 15.0) -> tuple[int, str, str]:
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate(timeout=5)
        pytest.fail(f"pipeline did not exit after SIGINT\nstdout={out}\nstderr={err}")
    return proc.returncode, out, err


def test_sigint_exits_130_without_partial_file_and_resume_skips_completed(
        slow_server, tmp_path) -> None:
    csv_path = tmp_path / "candidates.csv"
    ledger_path = tmp_path / "ledger.jsonl"
    raw_path = tmp_path / "raw"
    _write_csv(csv_path, slow_server.base_url)

    proc = _start_bulk(csv_path, ledger_path, raw_path)
    try:
        assert slow_server.started.wait(timeout=10), "slow download never started"
        time.sleep(0.1)
        proc.send_signal(signal.SIGINT)
        code, out, err = _finish(proc)
    finally:
        slow_server.release.set()

    assert code == 130, (out, err)
    assert "interrupted: SIGINT" in out
    assert "completed: 1" in out
    assert "remaining: 1" in out

    fast_checksum = hashlib.sha256(slow_server.fast_pdf).hexdigest()
    slow_checksum = hashlib.sha256(slow_server.slow_pdf).hexdigest()
    assert (raw_path / f"{fast_checksum}.pdf").is_file()
    assert not (raw_path / f"{slow_checksum}.pdf").exists()
    assert len(list(raw_path.glob("*.pdf"))) == 1

    ledger = Ledger(ledger_path)
    by_url = {c.source_url: c.state for c in ledger.candidates.values()}
    assert by_url[f"{slow_server.base_url}/fast.pdf"] == CandidateState.EXTRACTED
    assert by_url[f"{slow_server.base_url}/slow.pdf"] == CandidateState.FETCH_PENDING

    slow_server.fast = True
    proc2 = _start_bulk(csv_path, ledger_path, raw_path)
    code2, out2, err2 = _finish(proc2)
    assert code2 == 0, (out2, err2)
    assert "resumed: 1" in out2
    assert "extracted: 1" in out2
    assert "accounting invariant: PASS" in out2

    assert slow_server.hits["/fast.pdf"] == 1
    assert slow_server.hits["/slow.pdf"] == 2
    ledger2 = Ledger(ledger_path)
    assert len(ledger2.rows) == 2
    by_url2 = {c.source_url: c.state for c in ledger2.candidates.values()}
    assert by_url2[f"{slow_server.base_url}/fast.pdf"] == CandidateState.EXTRACTED
    assert by_url2[f"{slow_server.base_url}/slow.pdf"] == CandidateState.EXTRACTED
    assert (raw_path / f"{slow_checksum}.pdf").is_file()


def test_second_sigint_force_exits_without_traceback(slow_server, tmp_path) -> None:
    csv_path = tmp_path / "candidates.csv"
    ledger_path = tmp_path / "ledger.jsonl"
    raw_path = tmp_path / "raw"
    _write_csv(csv_path, slow_server.base_url)

    proc = _start_bulk(csv_path, ledger_path, raw_path)
    try:
        assert slow_server.started.wait(timeout=10), "slow download never started"
        time.sleep(0.1)
        proc.send_signal(signal.SIGINT)
        time.sleep(0.02)
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        code, out, err = _finish(proc)
    finally:
        slow_server.release.set()

    assert code == 130, (out, err)
    assert "Traceback" not in err
    slow_checksum = hashlib.sha256(slow_server.slow_pdf).hexdigest()
    assert not (raw_path / f"{slow_checksum}.pdf").exists()


def test_sigint_guard_second_signal_force_exits(monkeypatch) -> None:
    from src.ingestion.pipeline import EXIT_INTERRUPTED, _SigintGuard

    guard = _SigintGuard()
    guard._count = 1

    def force_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr("src.ingestion.pipeline.os._exit", force_exit)
    with pytest.raises(SystemExit) as exc:
        guard._handle(signal.SIGINT, None)
    assert exc.value.code == EXIT_INTERRUPTED
