"""Unit tests for LLD-38/39 fetcher behaviour.

All HTTP is served by a local stdlib server on 127.0.0.1. No external network.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest

from src.ingestion.fetcher import Fetcher, FetchError

PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
HTML = b"<!DOCTYPE html><html><body>not a pdf</body></html>"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        server = self.server
        script = getattr(server, "script", [])
        spec = script.pop(0) if script else (500, {}, b"")
        status, headers, body = spec
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        if "Content-Length" not in headers:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args, **kwargs) -> None:
        return


@pytest.fixture()
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.script = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _url(server: ThreadingHTTPServer) -> str:
    address = cast("tuple[str, int]", server.server_address)
    return f"http://{address[0]}:{address[1]}/filing.pdf"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_retry_after_seconds_is_honoured(http_server) -> None:
    http_server.script = [
        (429, {"Retry-After": "5"}, b"slow down"),
        (200, {"Content-Type": "application/pdf"}, PDF),
    ]
    fake = _Clock()
    fetcher = Fetcher(
        min_interval_seconds=0.0,
        max_retries=3,
        sleep=fake.sleep,
        clock=fake.clock,
    )
    data, _attempt = fetcher.fetch(_url(http_server), "c1")
    assert data == PDF
    assert fake.sleeps == [5.0]


def test_retry_after_http_date_is_honoured(http_server) -> None:
    now = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
    retry_at = now + timedelta(seconds=10)
    http_server.script = [
        (429, {"Retry-After": format_datetime(retry_at, usegmt=True)}, b"slow down"),
        (200, {"Content-Type": "application/pdf"}, PDF),
    ]
    fake = _Clock()
    fetcher = Fetcher(
        min_interval_seconds=0.0,
        max_retries=3,
        sleep=fake.sleep,
        clock=fake.clock,
        now=lambda: now,
    )
    data, _attempt = fetcher.fetch(_url(http_server), "c1")
    assert data == PDF
    assert fake.sleeps == [10.0]


def test_exponential_backoff_is_capped(http_server) -> None:
    http_server.script = [(503, {}, b"busy") for _ in range(3)]
    fake = _Clock()
    fetcher = Fetcher(
        min_interval_seconds=0.0,
        max_retries=3,
        backoff_base=10.0,
        backoff_cap_seconds=15.0,
        sleep=fake.sleep,
        clock=fake.clock,
    )
    with pytest.raises(FetchError) as exc:
        fetcher.fetch(_url(http_server), "c1")
    assert exc.value.code == "http_503"
    assert exc.value.retryable is True
    assert fake.sleeps == [10.0, 15.0]


def test_permanent_4xx_is_not_retried(http_server) -> None:
    http_server.script = [(404, {}, b"missing")]
    fake = _Clock()
    fetcher = Fetcher(
        min_interval_seconds=0.0,
        max_retries=3,
        sleep=fake.sleep,
        clock=fake.clock,
    )
    with pytest.raises(FetchError) as exc:
        fetcher.fetch(_url(http_server), "c1")
    assert exc.value.code == "http_404"
    assert exc.value.retryable is False
    assert fake.sleeps == []


def test_retryable_status_exhaustion_keeps_attempt_bound(http_server) -> None:
    http_server.script = [(500, {}, b"boom") for _ in range(2)]
    fake = _Clock()
    fetcher = Fetcher(
        min_interval_seconds=0.0,
        max_retries=2,
        backoff_base=2.0,
        sleep=fake.sleep,
        clock=fake.clock,
    )
    with pytest.raises(FetchError) as exc:
        fetcher.fetch(_url(http_server), "c1")
    assert exc.value.code == "http_500"
    assert exc.value.retryable is True
    assert fake.sleeps == [2.0]


class _FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status_code = status
        self.headers = {"Content-Length": str(len(body))}
        self._body = body
        self.closed = False

    def iter_content(self, chunk_size: int = 65536):
        yield self._body

    def close(self) -> None:
        self.closed = True


def test_streamed_response_is_closed_on_success(monkeypatch) -> None:
    response = _FakeResponse(200, PDF)
    monkeypatch.setattr("src.ingestion.fetcher.requests.get", lambda *a, **k: response)
    fetcher = Fetcher(min_interval_seconds=0.0, sleep=lambda _s: None, clock=lambda: 0.0)
    data, _attempt = fetcher.fetch("http://example.test/a.pdf", "c1")
    assert data == PDF
    assert response.closed is True


def test_streamed_response_is_closed_on_failure(monkeypatch) -> None:
    response = _FakeResponse(200, b"<html>not a pdf</html>")
    monkeypatch.setattr("src.ingestion.fetcher.requests.get", lambda *a, **k: response)
    fetcher = Fetcher(min_interval_seconds=0.0, sleep=lambda _s: None, clock=lambda: 0.0)
    with pytest.raises(FetchError):
        fetcher.fetch("http://example.test/a.pdf", "c1")
    assert response.closed is True


def test_html_body_at_pdf_url_is_html_only(http_server) -> None:
    http_server.script = [
        (200, {"Content-Type": "text/html; charset=utf-8"}, HTML),
    ]
    fake = _Clock()
    fetcher = Fetcher(
        min_interval_seconds=0.0,
        max_retries=2,
        sleep=fake.sleep,
        clock=fake.clock,
    )
    with pytest.raises(FetchError) as exc:
        fetcher.fetch(_url(http_server), "c1")
    assert exc.value.code == "html_only"
    assert exc.value.retryable is False
    assert exc.value.attempt is not None
    assert exc.value.attempt.error_code == "html_only"
    assert fake.sleeps == []


def test_short_content_length_is_retried_then_succeeds(http_server) -> None:
    http_server.script = [
        (200, {"Content-Length": str(len(PDF) + 10), "Connection": "close"}, PDF),
        (200, {"Content-Type": "application/pdf"}, PDF),
    ]
    fake = _Clock()
    fetcher = Fetcher(
        min_interval_seconds=0.0,
        max_retries=2,
        backoff_base=2.0,
        sleep=fake.sleep,
        clock=fake.clock,
    )
    data, attempt = fetcher.fetch(_url(http_server), "c1")
    assert data == PDF
    assert attempt.attempt_number == 2
    assert fake.sleeps == [2.0]


def test_truncated_body_exhaustion_is_corrupt_pdf(http_server) -> None:
    http_server.script = [
        (200, {"Content-Length": str(len(PDF) + 10), "Connection": "close"}, PDF[:10]),
        (200, {"Content-Length": str(len(PDF) + 10), "Connection": "close"}, PDF[:10]),
    ]
    fake = _Clock()
    fetcher = Fetcher(
        min_interval_seconds=0.0,
        max_retries=2,
        backoff_base=2.0,
        sleep=fake.sleep,
        clock=fake.clock,
    )
    with pytest.raises(FetchError) as exc:
        fetcher.fetch(_url(http_server), "c1")
    assert exc.value.code == "corrupt_pdf"
    assert exc.value.retryable is True
    assert exc.value.attempt is not None
    assert exc.value.attempt.error_code == "truncated"
    assert fake.sleeps == [2.0]


def test_zero_byte_body_is_retried_then_succeeds(http_server) -> None:
    http_server.script = [
        (200, {"Content-Type": "application/pdf"}, b""),
        (200, {"Content-Type": "application/pdf"}, PDF),
    ]
    fake = _Clock()
    fetcher = Fetcher(
        min_interval_seconds=0.0,
        max_retries=2,
        backoff_base=2.0,
        sleep=fake.sleep,
        clock=fake.clock,
    )
    data, attempt = fetcher.fetch(_url(http_server), "c1")
    assert data == PDF
    assert attempt.attempt_number == 2
    assert fake.sleeps == [2.0]


def test_html_never_reaches_storage(http_server, tmp_path) -> None:
    from src.ingestion.ledger import Ledger
    from src.ingestion.models import Candidate, CandidateState, DocType, ExclusionReason
    from src.ingestion.pipeline import process_candidate
    from src.ingestion.storage import LocalRawStorage

    http_server.script = [
        (200, {"Content-Type": "text/html"}, HTML),
    ]
    fetcher = Fetcher(min_interval_seconds=0.0, max_retries=1, sleep=lambda _s: None,
                      clock=lambda: 0.0)
    ledger = Ledger()
    storage = LocalRawStorage(tmp_path)
    candidate = Candidate(
        candidate_id="c1", company_id="C", fiscal_year="FY25",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url=_url(http_server),
    )
    state = process_candidate(candidate, fetcher, ledger, storage)
    assert state == CandidateState.FETCH_FAILED.value
    assert list(tmp_path.rglob("*.pdf")) == []
    assert len(ledger.exclusions) == 1
    assert ledger.exclusions[0].reason == ExclusionReason.HTML_ONLY
    assert candidate.fetch_attempts[-1].error_code == "html_only"


def test_truncated_body_never_reaches_storage(http_server, tmp_path) -> None:
    from src.ingestion.ledger import Ledger
    from src.ingestion.models import Candidate, CandidateState, DocType, ExclusionReason
    from src.ingestion.pipeline import process_candidate
    from src.ingestion.storage import LocalRawStorage

    http_server.script = [
        (200, {"Content-Length": str(len(PDF) + 10), "Connection": "close"}, PDF[:10]),
        (200, {"Content-Length": str(len(PDF) + 10), "Connection": "close"}, PDF[:10]),
    ]
    fetcher = Fetcher(min_interval_seconds=0.0, max_retries=2, sleep=lambda _s: None,
                      clock=lambda: 0.0)
    ledger = Ledger()
    storage = LocalRawStorage(tmp_path)
    candidate = Candidate(
        candidate_id="c1", company_id="C", fiscal_year="FY25",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url=_url(http_server),
    )
    state = process_candidate(candidate, fetcher, ledger, storage)
    assert state == CandidateState.FETCH_FAILED.value
    assert list(tmp_path.rglob("*.pdf")) == []
    assert ledger.exclusions[0].reason == ExclusionReason.CORRUPT_PDF
    assert len(candidate.fetch_attempts) == 2
    assert candidate.fetch_attempts[-1].error_code == "truncated"
