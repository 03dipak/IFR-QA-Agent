"""Reconcile drift states: one test per state + idempotency (scratch dirs only)."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfWriter

from src.ingestion.dedupe import filing_id
from src.ingestion.extractor import extract_pages
from src.ingestion.ledger import Ledger
from src.ingestion.models import (
    Candidate,
    CandidateState,
    DocType,
    DocumentLedgerRow,
    ExclusionReason,
    ExtractionStatus,
    FetchAttempt,
)
from src.ingestion.reconcile import _model_bytes, main
from src.tables.extractor import extract_tables


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _make_doc(corpus: Path, ledger_path: Path, data: bytes, *,
              with_raw: bool = True, with_extracts: bool = True,
              page_count: int = 1) -> str:
    checksum = hashlib.sha256(data).hexdigest()
    extract_dir = corpus / "extract"
    if with_extracts:
        pages = extract_pages(data, document_id=checksum, extraction_run_id="test")
        tables = extract_tables(pages, document_id=checksum,
                                extraction_run_id="test", extraction_config_hash="cfg")
        (extract_dir / "page_extracts").mkdir(parents=True, exist_ok=True)
        (extract_dir / "table_records").mkdir(parents=True, exist_ok=True)
        (extract_dir / "page_extracts" / f"{checksum}.pages.jsonl").write_bytes(
            _model_bytes(pages))
        (extract_dir / "table_records" / f"{checksum}.tables.jsonl").write_bytes(
            _model_bytes(tables))
    if with_raw:
        (corpus / "raw").mkdir(parents=True, exist_ok=True)
        (corpus / "raw" / f"{checksum}.pdf").write_bytes(data)
    ledger = Ledger(ledger_path)
    ledger.register_document(DocumentLedgerRow(
        document_id=checksum,
        filing_id=filing_id("CO", "FY2025", "annual", DocType.ANNUAL.value, None),
        doc_type=DocType.ANNUAL, company_id="CO", fiscal_year="FY2025",
        reporting_period="annual",
        publication_date=datetime.now(UTC),
        page_count=page_count, checksum=checksum,
        extraction_status=ExtractionStatus.EXTRACTED,
        extraction_config_hash="cfg",
    ))
    return checksum


def _ledger_rows(ledger_path: Path) -> int:
    return sum(1 for line in ledger_path.read_text().splitlines() if line.strip())


def run(corpus: Path, ledger_path: Path, *args: str) -> int:
    return main(["--storage-dir", str(corpus), "--ledger", str(ledger_path), *args])


# a) raw + ledger row + extracts present; dry run writes nothing
def test_ok_state_and_dry_run_writes_nothing(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    ledger = tmp_path / "ledger.jsonl"
    _make_doc(corpus, ledger, _pdf_bytes())
    before = _tree_snapshot(tmp_path)
    assert run(corpus, ledger) == 0
    assert "OK" in capsys.readouterr().out
    assert _tree_snapshot(tmp_path) == before


# b) raw + ledger row, extracts missing
def test_missing_extract_then_re_extract_repairs(tmp_path):
    corpus = tmp_path / "corpus"
    ledger = tmp_path / "ledger.jsonl"
    checksum = _make_doc(corpus, ledger, _pdf_bytes(), with_extracts=False)
    assert run(corpus, ledger) == 2
    assert run(corpus, ledger, "--re-extract") == 0
    pages_path = corpus / "extract" / "page_extracts" / f"{checksum}.pages.jsonl"
    line = json.loads(pages_path.read_text().splitlines()[0])
    assert line["document_id"] == checksum


# c) raw present, no ledger row (orphan); second registration is a no-op
def test_orphan_registered_once(tmp_path):
    corpus = tmp_path / "corpus"
    ledger_path = tmp_path / "ledger.jsonl"
    data = _pdf_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    (corpus / "raw").mkdir(parents=True)
    (corpus / "raw" / f"{checksum}.pdf").write_bytes(data)
    assert run(corpus, ledger_path) == 2
    assert run(corpus, ledger_path, "--register-orphans") == 0
    row = Ledger(ledger_path).by_checksum(checksum)
    assert row is not None
    assert row.source == "local"
    assert row.source_url is None
    rows = _ledger_rows(ledger_path)
    # re-run: no duplicate rows; the new row now reports MISSING_EXTRACT (exit 2,
    # repairable with --re-extract like every pending row)
    assert run(corpus, ledger_path, "--register-orphans") == 2
    assert _ledger_rows(ledger_path) == rows


# d) ledger row present, raw PDF missing (fresh clone with ledger kept)
def test_missing_raw_reported_only(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    ledger = tmp_path / "ledger.jsonl"
    _make_doc(corpus, ledger, _pdf_bytes(), with_raw=False)
    assert run(corpus, ledger) == 2
    assert "MISSING_RAW" in capsys.readouterr().out
    assert run(corpus, ledger, "--re-extract") == 2  # never silently repaired
    assert "MISSING_RAW" in capsys.readouterr().out


# e) extract file corrupt, rebuilt idempotently
def test_corrupt_extract_rebuild_idempotent(tmp_path):
    corpus = tmp_path / "corpus"
    ledger = tmp_path / "ledger.jsonl"
    checksum = _make_doc(corpus, ledger, _pdf_bytes())
    pages_path = corpus / "extract" / "page_extracts" / f"{checksum}.pages.jsonl"
    pages_path.write_text('{"document_id": "x", "partial')
    tables_path = corpus / "extract" / "table_records" / f"{checksum}.tables.jsonl"
    tables_path.write_text('{"partial no newline')
    assert run(corpus, ledger, "--re-extract") == 0
    assert json.loads(pages_path.read_text().splitlines()[0])["document_id"] == checksum
    before, rows = _tree_snapshot(tmp_path), _ledger_rows(ledger)
    assert run(corpus, ledger, "--re-extract") == 0  # second run: no-op
    assert _tree_snapshot(tmp_path) == before
    assert _ledger_rows(ledger) == rows


# e2) truncated last line counts as corrupt
def test_truncated_line_is_corrupt(tmp_path):
    corpus = tmp_path / "corpus"
    ledger = tmp_path / "ledger.jsonl"
    _make_doc(corpus, ledger, _pdf_bytes(), page_count=1)
    pages_path = next((corpus / "extract" / "page_extracts").glob("*.pages.jsonl"))
    data = pages_path.read_text()
    pages_path.write_text(data[: len(data) // 2])  # drops the trailing newline
    assert run(corpus, ledger) == 2


# f) raw file whose checksum no longer matches its name; rebuild refuses
def test_checksum_mismatch_rebuild_refuses(tmp_path):
    corpus = tmp_path / "corpus"
    ledger = tmp_path / "ledger.jsonl"
    checksum = _make_doc(corpus, ledger, _pdf_bytes(), with_extracts=False)
    tampered = _pdf_bytes() + b"tampered"
    (corpus / "raw" / f"{checksum}.pdf").write_bytes(tampered)
    assert run(corpus, ledger) == 2
    assert run(corpus, ledger, "--re-extract") == 2  # refused: raw not intact


def _make_excluded_orphan(tmp_path, data: bytes) -> tuple[Path, Path, str]:
    corpus = tmp_path / "corpus"
    ledger_path = tmp_path / "ledger.jsonl"
    (corpus / "raw").mkdir(parents=True)
    checksum = hashlib.sha256(data).hexdigest()
    (corpus / "raw" / f"{checksum}.pdf").write_bytes(data)
    ledger = Ledger(ledger_path)
    attempt = FetchAttempt(
        fetch_attempt_id="a1", candidate_id="c1",
        source_url="https://x/y.pdf",
        downloaded_bytes=len(data), checksum=checksum)
    ledger.register_candidate(Candidate(
        candidate_id="c1", company_id="CO", fiscal_year="Q1FY2027",
        reporting_period="quarterly", doc_type=DocType.QUARTERLY,
        source_url="https://x/y.pdf",
        fetch_attempts=[attempt]))
    ledger.set_candidate_state(
        "c1", CandidateState.EXCLUDED,
        exclusion_reason=ExclusionReason.SCANNED_PDF)
    return corpus, ledger_path, checksum


# g) orphan whose bytes belong to an excluded candidate is refused,
#    unless explicitly allowed
def test_register_orphans_refuses_excluded_checksum(tmp_path, capsys):
    corpus, ledger_path, checksum = _make_excluded_orphan(
        tmp_path, _pdf_bytes())
    assert run(corpus, ledger_path, "--register-orphans") == 2
    out = capsys.readouterr().out
    assert "CONFLICT" in out
    assert "c1" in out
    assert Ledger(ledger_path).by_checksum(checksum) is None


def test_register_orphans_allow_excluded_registers(tmp_path, capsys):
    corpus, ledger_path, checksum = _make_excluded_orphan(
        tmp_path, _pdf_bytes())
    assert run(corpus, ledger_path,
               "--register-orphans", "--allow-excluded-orphans") == 0
    out = capsys.readouterr().out
    assert "CONFLICT" not in out
    row = Ledger(ledger_path).by_checksum(checksum)
    assert row is not None
    assert row.source == "local"


# h) one unextractable PDF must not abort --re-extract for the rest
def test_re_extract_continues_after_extract_error(tmp_path, monkeypatch, capsys):
    from src.ingestion import reconcile as reconcile_mod

    good = _make_doc(tmp_path / "corpus", tmp_path / "ledger.jsonl",
                     _pdf_bytes(), with_extracts=False)
    other_data = _pdf_bytes() + bytes([1])
    other = _make_doc(tmp_path / "corpus", tmp_path / "ledger.jsonl",
                      other_data, with_extracts=False)
    corpus = tmp_path / "corpus"
    ledger_path = tmp_path / "ledger.jsonl"

    real_extract_pages = reconcile_mod.extract_pages

    def boom(data, document_id, extraction_run_id):
        if document_id == other:
            raise ValueError(
                "extract_error: Reached maximum number of tokens per stream.")
        return real_extract_pages(data, document_id=document_id,
                                  extraction_run_id=extraction_run_id)

    monkeypatch.setattr(reconcile_mod, "extract_pages", boom)
    assert run(corpus, ledger_path, "--re-extract") == 2
    out = capsys.readouterr().out
    assert "EXTRACT_ERROR" in out
    good_pages = corpus / "extract" / "page_extracts" / f"{good}.pages.jsonl"
    assert json.loads(good_pages.read_text().splitlines()[0])["document_id"] == good
    assert not (corpus / "extract" / "page_extracts" / f"{other}.pages.jsonl").exists()
