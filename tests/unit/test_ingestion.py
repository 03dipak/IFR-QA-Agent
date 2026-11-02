"""Unit tests for LLD-01 ingestion pipeline."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from src.ingestion.checksummer import sha256_bytes
from src.ingestion.dedupe import classify_mirror, filing_id
from src.ingestion.extractor import text_density_ok
from src.ingestion.fetcher import Fetcher, FetchError
from src.ingestion.freeze_manager import FreezeManager
from src.ingestion.ledger import Ledger
from src.ingestion.models import (
    Candidate,
    CandidateState,
    DocType,
    DocumentLedgerRow,
    ExclusionReason,
    ExclusionRecord,
    FetchAttempt,
)
from src.ingestion.storage import LocalRawStorage, StorageConflictError


def _make_text_pdf(pages: list[list[str]]) -> bytes:
    """Build a deterministic native-text PDF, one list of lines per page."""
    from io import BytesIO

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
        })
        ops = ["BT", "/F1 12 Tf", "72 720 Td"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.extend((f"({escaped}) Tj", "0 -16 Td"))
        ops.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_pdf(
    text: str = "Hello world revenue 100 total assets 200 liabilities 50 "
                 "cash flow 25",
) -> bytes:
    return _make_text_pdf([[text]])


def test_sha256_bytes_deterministic():
    data = b"abc"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_filing_id_deterministic():
    a = filing_id("C1", "FY25", "Q4", "annual", "ANN1")
    b = filing_id("C1", "FY25", "Q4", "annual", "ANN1")
    assert a == b
    c = filing_id("C1", "FY25", "Q4", "annual", "ANN2")
    assert a != c


def test_mirror_exact_byte():
    data = b"same bytes"
    ledger = Ledger()
    # seed an existing row with same checksum
    from datetime import datetime

    from src.ingestion.models import AuditStatus, DocType, ExtractionStatus
    row = DocumentLedgerRow(
        document_id="d1", filing_id="f1", doc_type=DocType.ANNUAL,
        company_id="C1", fiscal_year="FY25", reporting_period="FY",
        publication_date=datetime.now(UTC), page_count=1,
        checksum=sha256_bytes(data), extraction_status=ExtractionStatus.EXTRACTED,
        extraction_config_hash="cfg", audit_status=AuditStatus.UNCLEAR,
    )
    ledger.register_document(row)
    match = classify_mirror(data, None, list(ledger.rows.values()))
    assert match.kind == "exact_byte"
    assert match.existing_document_id == "d1"


def test_ledger_idempotent_register():
    data = b"hello"
    ledger = Ledger()
    from datetime import datetime

    from src.ingestion.models import AuditStatus, DocType, ExtractionStatus
    row = DocumentLedgerRow(
        document_id="d1", filing_id="f1", doc_type=DocType.ANNUAL,
        company_id="C1", fiscal_year="FY25", reporting_period="FY",
        publication_date=datetime.now(UTC), page_count=1,
        checksum=sha256_bytes(data), extraction_status=ExtractionStatus.EXTRACTED,
        extraction_config_hash="cfg", audit_status=AuditStatus.UNCLEAR,
    )
    created1, did1 = ledger.register_document(row)
    assert created1 is True
    row2 = row.model_copy(update={"document_id": "d2"})
    created2, did2 = ledger.register_document(row2)
    assert created2 is False
    assert did2 == did1


def test_ledger_reconcile():
    ledger = Ledger()
    c1 = Candidate(candidate_id="c1", company_id="C1", fiscal_year="FY25",
                   reporting_period="FY", doc_type=DocType.ANNUAL,
                   source_url="http://x")
    ledger.register_candidate(c1)
    ledger.set_candidate_state("c1", CandidateState.FROZEN)
    s = ledger.reconcile(["c1"])
    assert s["discovered"] == 1
    assert s["accepted"] == 1
    assert s["unresolved"] == 0


def test_candidate_from_url_derives_candidate_id():
    from src.ingestion.pipeline import _candidate_from_url

    url = ("https://www.bseindia.com/xml-data/corpfiling/AttachHis/"
           "REL_500325_FY2024_ANNUAL.pdf")
    c = _candidate_from_url(url)
    assert c.source_url == url
    assert c.candidate_id.startswith("url-")
    assert _candidate_from_url(url).candidate_id == c.candidate_id


def test_storage_roundtrip(tmp_path):
    s = LocalRawStorage(tmp_path)
    data = b"pdf bytes"
    cs = s.put("a/b.pdf", data)
    assert cs == sha256_bytes(data)
    assert s.exists("a/b.pdf")
    assert s.get("a/b.pdf") == data
    s.delete("a/b.pdf")
    assert not s.exists("a/b.pdf")


def test_storage_path_traversal(tmp_path):
    s = LocalRawStorage(tmp_path)
    with pytest.raises(ValueError):
        s.get("../etc/passwd")


def test_freeze_manifest_checksum():
    from datetime import datetime

    from src.ingestion.models import AuditStatus, DocType, ExtractionStatus, PageExtract
    fm = FreezeManager()
    row = DocumentLedgerRow(
        document_id="d1", filing_id="f1", doc_type=DocType.ANNUAL,
        company_id="C1", fiscal_year="FY25", reporting_period="FY",
        publication_date=datetime.now(UTC), page_count=1,
        checksum="abc", extraction_status=ExtractionStatus.EXTRACTED,
        extraction_config_hash="cfg", audit_status=AuditStatus.UNCLEAR,
    )
    pe = PageExtract(document_id="d1", extraction_run_id="r1",
                     pdf_page_index=1, page_text="hi", extraction_hash="h")
    m = fm.create_freeze("v1", None, [row], [pe], "cfg", "run1", "chash",
                         qa_report_checksum="q", qa_passed=True)
    from src.ingestion.freeze_manager import CandidateAccounting
    fm.validate_freeze(m.freeze_id, CandidateAccounting(discovered=1, accepted=1,
                       permanently_excluded=0, unresolved=0), [], [pe])
    assert m.manifest_checksum is not None
    assert fm.verify(m.freeze_id)
    fm.publish(m.freeze_id)
    assert m.status == "published"


def test_text_density_ok():
    assert text_density_ok("a" * 50) is True
    assert text_density_ok("hi") is False


def _statement_candidate(candidate_id: str, url: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        company_id="LT",
        fiscal_year="FY2024",
        reporting_period="annual",
        doc_type=DocType.ANNUAL,
        source_url=url,
    )


def test_native_text_pdf_registers_document(tmp_path):
    from src.ingestion.pipeline import process_candidate

    pdf = _make_text_pdf([[
        "Balance Sheet",
        "Total assets 100 revenue 200 liabilities 50 cash flow 10",
    ]])
    ledger = Ledger()
    candidate = _statement_candidate("native-1", "https://ok/native.pdf")
    state = process_candidate(
        candidate, FakeFetcher({"https://ok/native.pdf": pdf}),
        ledger, LocalRawStorage(tmp_path))
    assert state == CandidateState.EXTRACTED.value
    assert len(ledger.rows) == 1
    assert ledger.exclusions == []
    assert candidate.state == CandidateState.EXTRACTED


def test_scanned_statement_pdf_excludes_whole_document(tmp_path):
    from src.ingestion.pipeline import process_candidate

    pdf = _make_text_pdf([[]])
    ledger = Ledger()
    candidate = _statement_candidate("scan-1", "https://ok/scan.pdf")
    state = process_candidate(
        candidate, FakeFetcher({"https://ok/scan.pdf": pdf}),
        ledger, LocalRawStorage(tmp_path))
    assert state == CandidateState.EXCLUDED.value
    assert ledger.rows == {}
    assert len(ledger.exclusions) == 1
    assert ledger.exclusions[0].reason == ExclusionReason.SCANNED_PDF
    stored = ledger.candidates["scan-1"]
    assert stored.state == CandidateState.EXCLUDED
    assert stored.exclusion_reason == ExclusionReason.SCANNED_PDF


def test_mixed_native_and_scanned_pdf_excludes_whole_document(tmp_path):
    from src.ingestion.pipeline import process_candidate

    pdf = _make_text_pdf([
        ["Cover page with enough native prose text to pass the density test "
         "and not be classified as a scan"],
        [],
    ])
    ledger = Ledger()
    candidate = _statement_candidate("mixed-1", "https://ok/mixed.pdf")
    state = process_candidate(
        candidate, FakeFetcher({"https://ok/mixed.pdf": pdf}),
        ledger, LocalRawStorage(tmp_path))
    assert state == CandidateState.EXCLUDED.value
    assert ledger.rows == {}
    assert ledger.exclusions[0].reason == ExclusionReason.SCANNED_PDF


def test_garbled_statement_text_excludes_whole_document(tmp_path, monkeypatch):
    from src.ingestion import pipeline as pipeline_module
    from src.ingestion.models import PageExtract, PageType
    from src.ingestion.pipeline import process_candidate

    pdf = _make_text_pdf([[
        "Balance Sheet total assets 100 revenue 200 liabilities 50",
    ]])

    def fake_extract(_data, document_id, run_id):
        return [PageExtract(
            document_id=document_id,
            extraction_run_id=run_id,
            pdf_page_index=1,
            page_text="\ufffd" * 80,
            extraction_hash="garbled",
            page_type=PageType.STATEMENT,
            character_count=80,
        )]

    monkeypatch.setattr(pipeline_module, "extract_pages", fake_extract)
    ledger = Ledger()
    candidate = _statement_candidate("garbled-1", "https://ok/garbled.pdf")
    state = process_candidate(
        candidate, FakeFetcher({"https://ok/garbled.pdf": pdf}),
        ledger, LocalRawStorage(tmp_path))
    assert state == CandidateState.EXCLUDED.value
    assert ledger.rows == {}
    assert ledger.exclusions[0].reason == ExclusionReason.SCANNED_PDF


class FakeFetcher(Fetcher):
    def __init__(self, pages: dict[str, bytes]) -> None:
        super().__init__(min_interval_seconds=0.0)
        self.pages = pages

    def fetch(self, url: str, candidate_id: str) -> tuple[bytes, FetchAttempt]:
        if url not in self.pages:
            raise FetchError("not_found", "not found", retryable=False)
        data = self.pages[url]
        attempt = FetchAttempt(
            fetch_attempt_id="a1",
            candidate_id=candidate_id,
            source_url=url,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            http_status=200,
            content_length=len(data),
            downloaded_bytes=len(data),
            attempt_number=1,
        )
        return data, attempt


def test_candidate_from_discovery_row_maps_fields():
    from src.ingestion.pipeline import candidate_from_discovery_row

    c = candidate_from_discovery_row({
        "company_id": "LT", "scrip_code": "500834", "document_type": "annual",
        "fiscal_period": "FY2024", "filing_date": "2024-03-15",
        "url": "https://www.bseindia.com/x/LT_FY2024_AN.pdf",
        "source_page": "http://list", "discovered_at": "now",
    })
    assert c.source_url == "https://www.bseindia.com/x/LT_FY2024_AN.pdf"
    assert c.company_id == "LT"
    assert c.fiscal_year == "FY2024"
    assert c.reporting_period == "annual"
    assert c.doc_type.value == "annual"
    assert c.candidate_id.startswith("disc-")


def test_candidate_from_discovery_row_defaults_and_rejects_invalid_type():
    from src.ingestion.pipeline import candidate_from_discovery_row

    c = candidate_from_discovery_row({"url": "https://x/1.pdf"})
    assert c.doc_type.value == "annual"
    with pytest.raises(ValueError, match="invalid document_type"):
        candidate_from_discovery_row({"url": "https://x/1.pdf", "document_type": "bogus"})
    with pytest.raises(ValueError):
        candidate_from_discovery_row({"url": "", "company_id": "C"})


def test_load_discovery_candidates_reads_csv(tmp_path):
    from src.ingestion.pipeline import load_discovery_candidates

    p = tmp_path / "candidates.csv"
    p.write_text(
        "company_id,scrip_code,document_type,fiscal_period,filing_date,url,source_page,discovered_at\n"
        "LT,500834,annual,FY2024,2024-03-15,https://x/LT.pdf,http://list,now\n",
        encoding="utf-8",
    )
    cands = load_discovery_candidates(str(p))
    assert len(cands) == 1
    assert cands[0].source_url == "https://x/LT.pdf"


def test_run_bulk_processes_all_and_quarantines(tmp_path):
    from src.ingestion.pipeline import run_bulk

    fetcher = FakeFetcher({"https://ok/1.pdf": _make_pdf()})
    ledger = Ledger()
    storage = LocalRawStorage(tmp_path)
    good = Candidate(
        candidate_id="disc-good", company_id="LT", fiscal_year="FY2024",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url="https://ok/1.pdf",
    )
    bad = Candidate(
        candidate_id="disc-bad", company_id="LT", fiscal_year="FY2023",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url="https://bad/2.pdf",
    )
    counts, quarantined = run_bulk([good, bad], fetcher=fetcher, ledger=ledger, storage=storage)
    assert len(quarantined) == 1
    assert quarantined[0] == "disc-bad"
    assert "extracted" in counts or "fetched" in counts or "mirror" in counts


def test_reconcile_quarantine_matches_ledger():
    from src.ingestion.pipeline import reconcile_quarantine

    ledger = Ledger()
    ledger.add_exclusion(
        ExclusionRecord(
            candidate_id="disc-bad",
            reason=ExclusionReason.FETCH_FAILED,
            measurement="not_found",
        )
    )
    result = reconcile_quarantine(ledger, ["disc-bad"])
    assert result["quarantined_total"] == 1
    assert result["with_exclusion_record"] == 1
    assert result["unmatched_quarantined"] == 0

    ledger2 = Ledger()
    result2 = reconcile_quarantine(ledger2, ["disc-bad"])
    assert result2["quarantined_total"] == 1
    assert result2["with_exclusion_record"] == 0
    assert result2["unmatched_quarantined"] == 1


def test_load_discovery_candidates_reports_missing_header(tmp_path):
    from src.ingestion.pipeline import load_discovery_candidates

    p = tmp_path / "candidates.csv"
    p.write_text("company_id\nC\n", encoding="utf-8")
    loaded = load_discovery_candidates(str(p))
    assert len(loaded) == 0
    assert any("missing required column" in problem and "url" in problem
               for problem in loaded.problems)


def test_load_discovery_candidates_reports_empty_file(tmp_path):
    from src.ingestion.pipeline import load_discovery_candidates

    p = tmp_path / "candidates.csv"
    p.write_text("", encoding="utf-8")
    loaded = load_discovery_candidates(str(p))
    assert len(loaded) == 0
    assert any("empty file" in problem for problem in loaded.problems)


def test_load_discovery_candidates_reports_header_only(tmp_path):
    from src.ingestion.pipeline import load_discovery_candidates

    p = tmp_path / "candidates.csv"
    p.write_text("company_id,url\n", encoding="utf-8")
    loaded = load_discovery_candidates(str(p))
    assert len(loaded) == 0
    assert any("header-only" in problem for problem in loaded.problems)


def test_load_discovery_candidates_reports_duplicate_header(tmp_path):
    from src.ingestion.pipeline import load_discovery_candidates

    p = tmp_path / "candidates.csv"
    p.write_text("url,url\nhttps://x/1.pdf,https://x/2.pdf\n", encoding="utf-8")
    loaded = load_discovery_candidates(str(p))
    assert len(loaded) == 0
    assert any("duplicate header" in problem for problem in loaded.problems)


def test_load_discovery_candidates_reports_malformed_quoting(tmp_path):
    from src.ingestion.pipeline import load_discovery_candidates

    p = tmp_path / "candidates.csv"
    p.write_text('company_id,url\nC,"https://x/1.pdf\n', encoding="utf-8")
    loaded = load_discovery_candidates(str(p))
    assert len(loaded) == 0
    assert any("malformed CSV" in problem for problem in loaded.problems)


def test_load_discovery_candidates_skips_bad_rows_but_keeps_good_ones(tmp_path):
    from src.ingestion.pipeline import load_discovery_candidates

    p = tmp_path / "candidates.csv"
    p.write_text(
        "company_id,document_type,fiscal_period,url\n"
        "C1,annual,FY2024,https://x/1.pdf\n"
        "C2,annual,FY2024,\n"
        "C3,bogus,FY2024,https://x/3.pdf\n"
        "C4,quarterly,Q1FY25,https://x/4.pdf\n",
        encoding="utf-8",
    )
    loaded = load_discovery_candidates(str(p))
    assert [c.source_url for c in loaded] == ["https://x/1.pdf", "https://x/4.pdf"]
    assert loaded.rows_seen == 4
    assert any("line 3: missing url" in problem for problem in loaded.problems)
    assert any("line 4: invalid document_type" in problem for problem in loaded.problems)


def test_load_discovery_candidates_reports_field_count_mismatch(tmp_path):
    from src.ingestion.pipeline import load_discovery_candidates

    p = tmp_path / "candidates.csv"
    p.write_text(
        "company_id,url\n"
        "C1,https://x/1.pdf,extra\n"
        "C2,https://x/2.pdf\n",
        encoding="utf-8",
    )
    loaded = load_discovery_candidates(str(p))
    assert [c.source_url for c in loaded] == ["https://x/2.pdf"]
    assert any("expected 2 columns, got 3" in problem for problem in loaded.problems)


def test_bulk_main_reports_bad_csv_and_exits_two(tmp_path, capsys):
    from src.ingestion.pipeline import main

    p = tmp_path / "candidates.csv"
    p.write_text("company_id\nC\n", encoding="utf-8")
    assert main(["bulk", "--candidates", str(p)]) == 2
    out = capsys.readouterr().out
    assert "candidate CSV:" in out
    assert "missing required column" in out


def test_bulk_default_storage_layout_is_canonical(tmp_path):
    from src.ingestion.pipeline import run_bulk

    pdf = _make_pdf()
    fetcher = FakeFetcher({"https://ok/1.pdf": pdf})
    ledger = Ledger()
    root = tmp_path / "data" / "corpus" / "raw"
    storage = LocalRawStorage(root)
    candidate = Candidate(
        candidate_id="disc-canonical", company_id="LT", fiscal_year="FY2024",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url="https://ok/1.pdf",
    )
    run_bulk([candidate], fetcher=fetcher, ledger=ledger, storage=storage)
    files = list(root.rglob("*.pdf"))
    assert len(files) == 1
    assert files[0].parent == root
    assert files[0].name == f"{sha256_bytes(pdf)}.pdf"
    assert not (root / "raw").exists()


def test_bulk_overridden_storage_prefix(tmp_path):
    from src.ingestion.pipeline import process_candidate

    pdf = _make_pdf()
    fetcher = FakeFetcher({"https://ok/1.pdf": pdf})
    ledger = Ledger()
    storage = LocalRawStorage(tmp_path)
    candidate = Candidate(
        candidate_id="disc-prefix", company_id="LT", fiscal_year="FY2024",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url="https://ok/1.pdf",
    )
    process_candidate(candidate, fetcher, ledger, storage, storage_key_prefix="tenant")
    assert (tmp_path / "tenant" / f"{sha256_bytes(pdf)}.pdf").is_file()


def test_storage_reads_legacy_raw_prefix(tmp_path):
    pdf = b"legacy pdf bytes"
    checksum = hashlib.sha256(pdf).hexdigest()
    legacy = tmp_path / "raw" / f"{checksum}.pdf"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(pdf)

    storage = LocalRawStorage(tmp_path)
    assert storage.exists(f"{checksum}.pdf")
    assert storage.get(f"{checksum}.pdf") == pdf


def test_storage_put_is_idempotent_and_refuses_conflict(tmp_path):
    storage = LocalRawStorage(tmp_path)
    key = "abc.pdf"
    assert storage.put(key, b"first") == hashlib.sha256(b"first").hexdigest()
    assert storage.put(key, b"first") == hashlib.sha256(b"first").hexdigest()
    with pytest.raises(StorageConflictError):
        storage.put(key, b"second")
    assert storage.get(key) == b"first"


def test_storage_distinct_keys_do_not_overwrite(tmp_path):
    storage = LocalRawStorage(tmp_path)
    storage.put("one.pdf", b"one")
    storage.put("two.pdf", b"two")
    assert storage.get("one.pdf") == b"one"
    assert storage.get("two.pdf") == b"two"


def _document_row(pdf: bytes) -> DocumentLedgerRow:
    checksum = sha256_bytes(pdf)
    return DocumentLedgerRow(
        document_id=checksum,
        filing_id="filing-1",
        doc_type=DocType.ANNUAL,
        company_id="LT",
        fiscal_year="FY2024",
        reporting_period="annual",
        publication_date=datetime.now(UTC),
        page_count=1,
        checksum=checksum,
        extraction_config_hash="cfg",
    )


def test_ledger_persists_and_replays_document_rows(tmp_path):
    from src.ingestion.ledger import Ledger

    path = tmp_path / "ledger.jsonl"
    pdf = b"persisted pdf"
    row = _document_row(pdf)
    first = Ledger(path)
    assert first.register_document(row) == (True, row.document_id)
    assert path.is_file()

    second = Ledger(path)
    assert second.by_checksum(row.checksum) is not None
    assert second.by_checksum(row.checksum).document_id == row.document_id
    assert second.register_document(row) == (False, row.document_id)


def test_ledger_persists_candidate_state_and_exclusions(tmp_path):
    from src.ingestion.ledger import Ledger

    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    candidate = Candidate(
        candidate_id="disc-persist", company_id="LT", fiscal_year="FY2024",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url="https://x/1.pdf",
    )
    ledger.register_candidate(candidate)
    ledger.set_candidate_state("disc-persist", CandidateState.EXTRACTED,
                               document_id="doc-1")
    ledger.add_exclusion(ExclusionRecord(
        candidate_id="disc-persist", reason=ExclusionReason.SCANNED_PDF))

    reloaded = Ledger(path)
    assert reloaded.candidates["disc-persist"].state == CandidateState.EXTRACTED
    assert reloaded.candidates["disc-persist"].document_id == "doc-1"
    assert len(reloaded.exclusions) == 1
    assert reloaded.exclusions[0].reason == ExclusionReason.SCANNED_PDF


def test_ledger_second_run_mirrors_without_duplicate_document(tmp_path):
    from src.ingestion.ledger import Ledger
    from src.ingestion.pipeline import process_candidate

    pdf = _make_pdf()
    path = tmp_path / "ledger.jsonl"
    storage = LocalRawStorage(tmp_path / "raw")

    first_ledger = Ledger(path)
    first_candidate = Candidate(
        candidate_id="disc-run", company_id="LT", fiscal_year="FY2024",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url="https://ok/1.pdf",
    )
    first_state = process_candidate(
        first_candidate, FakeFetcher({"https://ok/1.pdf": pdf}),
        first_ledger, storage)
    assert first_state == "extracted"
    assert len(first_ledger.rows) == 1
    document_id = next(iter(first_ledger.rows))

    second_ledger = Ledger(path)
    second_candidate = Candidate(
        candidate_id="disc-run", company_id="LT", fiscal_year="FY2024",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url="https://ok/1.pdf",
    )
    second_state = process_candidate(
        second_candidate, FakeFetcher({"https://ok/1.pdf": pdf}),
        second_ledger, storage)
    assert second_state == "mirror"
    assert list(second_ledger.rows) == [document_id]
    assert second_ledger.by_checksum(sha256_bytes(pdf)).document_id == document_id
    assert second_ledger.candidates["disc-run"].state == CandidateState.FETCHED


def test_ledger_missing_file_is_empty_until_first_write(tmp_path):
    from src.ingestion.ledger import Ledger

    path = tmp_path / "absent.jsonl"
    ledger = Ledger(path)
    assert ledger.rows == {}
    assert not path.exists()
    ledger.add_exclusion(ExclusionRecord(
        candidate_id="c1", reason=ExclusionReason.FETCH_FAILED))
    assert path.is_file()


def test_ledger_rejects_corrupt_and_unsupported_records(tmp_path):
    from src.ingestion.ledger import Ledger, LedgerError

    path = tmp_path / "ledger.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="invalid JSON"):
        Ledger(path)

    path.write_text('{"schema": 99, "kind": "candidate", "data": {}}\n', encoding="utf-8")
    with pytest.raises(LedgerError, match="unsupported schema"):
        Ledger(path)

    path.write_text('{"schema": 1, "kind": "mystery", "data": {}}\n', encoding="utf-8")
    with pytest.raises(LedgerError, match="unknown record kind"):
        Ledger(path)


def test_ledger_default_path_is_project_local():
    from src.ingestion.ledger import DEFAULT_LEDGER_PATH

    assert DEFAULT_LEDGER_PATH == "data/corpus/ledger.jsonl"


def test_run_main_reports_corrupt_ledger_without_network(tmp_path, capsys):
    from src.ingestion.pipeline import main

    path = tmp_path / "ledger.jsonl"
    path.write_text("{bad}\n", encoding="utf-8")
    rc = main([
        "run", "--url", "https://example.invalid/x.pdf",
        "--ledger", str(path), "--storage-dir", str(tmp_path / "raw"),
    ])
    assert rc == 2
    assert "ledger error:" in capsys.readouterr().out


class _CountingFetcher(FakeFetcher):
    def __init__(self, pages: dict[str, bytes]) -> None:
        super().__init__(pages)
        self.calls: list[str] = []

    def fetch(self, url: str, candidate_id: str):
        self.calls.append(url)
        return super().fetch(url, candidate_id)


def _bulk_candidate(candidate_id, url, fiscal_year="FY2024",
                    doc_type=DocType.ANNUAL, announcement_id=None):
    return Candidate(
        candidate_id=candidate_id, company_id="LT", fiscal_year=fiscal_year,
        reporting_period="annual", doc_type=doc_type, source_url=url,
        announcement_id=announcement_id,
    )


def test_run_bulk_skips_repeated_url_before_fetch(tmp_path):
    from src.ingestion.pipeline import run_bulk

    pdf = _make_pdf()
    fetcher = _CountingFetcher({"https://ok/1.pdf": pdf})
    ledger = Ledger()
    storage = LocalRawStorage(tmp_path)
    first = _bulk_candidate("disc-a", "https://ok/1.pdf")
    second = _bulk_candidate("disc-a-copy", "https://ok/1.pdf")
    counts, quarantined = run_bulk(
        [first, second], fetcher=fetcher, ledger=ledger, storage=storage)
    assert fetcher.calls == ["https://ok/1.pdf"]
    assert counts["duplicate"] == 1
    assert len(ledger.rows) == 1
    assert quarantined == []


def test_run_bulk_skips_same_logical_filing_before_fetch(tmp_path):
    from src.ingestion.pipeline import run_bulk

    pdf = _make_pdf()
    fetcher = _CountingFetcher({
        "https://ok/1.pdf": pdf,
        "https://mirror/1.pdf": pdf + b"\n",
    })
    ledger = Ledger()
    storage = LocalRawStorage(tmp_path)
    first = _bulk_candidate("disc-a", "https://ok/1.pdf", announcement_id="ANN-1")
    near_duplicate = _bulk_candidate(
        "disc-b", "https://mirror/1.pdf", announcement_id="ANN-1")
    counts, quarantined = run_bulk(
        [first, near_duplicate], fetcher=fetcher, ledger=ledger, storage=storage)
    assert fetcher.calls == ["https://ok/1.pdf"]
    assert counts["duplicate"] == 1
    assert len(ledger.rows) == 1
    assert quarantined == []


def test_run_bulk_keeps_distinct_filings(tmp_path):
    from src.ingestion.pipeline import run_bulk

    pdf1 = _make_pdf()
    pdf2 = _make_pdf() + b"\n"
    fetcher = _CountingFetcher({
        "https://ok/1.pdf": pdf1,
        "https://ok/2.pdf": pdf2,
    })
    ledger = Ledger()
    storage = LocalRawStorage(tmp_path)
    first = _bulk_candidate("disc-a", "https://ok/1.pdf", fiscal_year="FY2024")
    second = _bulk_candidate("disc-b", "https://ok/2.pdf", fiscal_year="FY2023")
    counts, quarantined = run_bulk(
        [first, second], fetcher=fetcher, ledger=ledger, storage=storage)
    assert fetcher.calls == ["https://ok/1.pdf", "https://ok/2.pdf"]
    assert counts["duplicate"] == 0
    assert len(ledger.rows) == 2
    assert quarantined == []


def test_run_bulk_duplicate_rows_converge_to_mirror_on_rerun(tmp_path):
    from src.ingestion.ledger import Ledger
    from src.ingestion.pipeline import run_bulk

    pdf = _make_pdf()
    path = tmp_path / "ledger.jsonl"
    storage = LocalRawStorage(tmp_path / "raw")

    first_fetcher = _CountingFetcher({"https://ok/1.pdf": pdf})
    first_ledger = Ledger(path)
    first_counts, _ = run_bulk(
        [_bulk_candidate("disc-a", "https://ok/1.pdf"),
         _bulk_candidate("disc-a-copy", "https://ok/1.pdf")],
        fetcher=first_fetcher, ledger=first_ledger, storage=storage)
    assert first_counts == {"extracted": 1, "duplicate": 1}
    assert len(first_ledger.rows) == 1
    document_id = next(iter(first_ledger.rows))

    second_fetcher = _CountingFetcher({"https://ok/1.pdf": pdf})
    second_ledger = Ledger(path)
    second_counts, _ = run_bulk(
        [_bulk_candidate("disc-a", "https://ok/1.pdf"),
         _bulk_candidate("disc-a-copy", "https://ok/1.pdf")],
        fetcher=second_fetcher, ledger=second_ledger, storage=storage)
    assert second_counts == {"mirror": 1, "duplicate": 1}
    assert list(second_ledger.rows) == [document_id]


def _ledger_candidate(candidate_id: str = "c1") -> Candidate:
    return Candidate(
        candidate_id=candidate_id, company_id="LT", fiscal_year="FY2024",
        reporting_period="annual", doc_type=DocType.ANNUAL,
        source_url=f"https://x/{candidate_id}.pdf",
    )


def test_counts_reconcile_balanced_ledger():
    ledger = Ledger()
    ledger.register_candidate(_ledger_candidate())
    ledger.set_candidate_state("c1", CandidateState.EXTRACTED)
    accounting = ledger.accounting()
    assert accounting["discovered"] == 1
    assert accounting["accepted"] == 1
    assert accounting["permanently_excluded"] == 0
    assert accounting["unresolved"] == 0
    assert accounting["balanced"] is True
    assert accounting["ok"] is True
    assert ledger.counts_reconcile() is True


def test_counts_reconcile_fails_with_unresolved_candidate():
    ledger = Ledger()
    ledger.register_candidate(_ledger_candidate())
    accounting = ledger.accounting()
    assert accounting["unresolved"] == 1
    assert accounting["unresolved_candidate_ids"] == ["c1"]
    assert accounting["ok"] is False
    assert ledger.counts_reconcile() is False


def test_counts_reconcile_counts_excluded_candidate_once():
    ledger = Ledger()
    ledger.register_candidate(_ledger_candidate())
    ledger.set_candidate_state("c1", CandidateState.EXCLUDED,
                               exclusion_reason=ExclusionReason.SCANNED_PDF)
    ledger.add_exclusion(ExclusionRecord(
        candidate_id="c1", reason=ExclusionReason.SCANNED_PDF))
    accounting = ledger.accounting()
    assert accounting["accepted"] == 0
    assert accounting["permanently_excluded"] == 1
    assert accounting["unresolved"] == 0
    assert accounting["ok"] is True


def test_counts_reconcile_fetch_failure_with_reason_is_permanent():
    ledger = Ledger()
    ledger.register_candidate(_ledger_candidate())
    ledger.set_candidate_state("c1", CandidateState.FETCH_FAILED)
    ledger.add_exclusion(ExclusionRecord(
        candidate_id="c1", reason=ExclusionReason.HTML_ONLY))
    accounting = ledger.accounting()
    assert accounting["permanently_excluded"] == 1
    assert accounting["unresolved"] == 0
    assert accounting["ok"] is True


def test_counts_reconcile_fetch_failure_without_reason_is_unresolved():
    ledger = Ledger()
    ledger.register_candidate(_ledger_candidate())
    ledger.set_candidate_state("c1", CandidateState.FETCH_FAILED)
    assert ledger.counts_reconcile() is False


def test_counts_reconcile_counts_mirror_once_without_new_document():
    ledger = Ledger()
    ledger.register_candidate(_ledger_candidate())
    ledger.set_candidate_state("c1", CandidateState.FETCHED)
    accounting = ledger.accounting()
    assert accounting["mirrors"] == 1
    assert accounting["accepted"] == 0
    assert accounting["permanently_excluded"] == 1
    assert accounting["unresolved"] == 0
    assert accounting["ok"] is True


def test_counts_reconcile_includes_fetch_attempt_totals():
    now = datetime.now(UTC)
    candidate = _ledger_candidate()
    candidate.fetch_attempts = [
        FetchAttempt(
            fetch_attempt_id="a1", candidate_id="c1", source_url="https://x/c1.pdf",
            started_at=now, completed_at=now, http_status=503,
            downloaded_bytes=0, attempt_number=1,
            error_code="http_503", retryable=True,
        ),
        FetchAttempt(
            fetch_attempt_id="a2", candidate_id="c1", source_url="https://x/c1.pdf",
            started_at=now, completed_at=now, http_status=200,
            downloaded_bytes=100, attempt_number=2,
        ),
    ]
    ledger = Ledger()
    ledger.register_candidate(candidate)
    ledger.set_candidate_state("c1", CandidateState.EXTRACTED)
    accounting = ledger.accounting()
    assert accounting["fetch_attempts"] == 2
    assert accounting["successful_fetches"] == 1
    assert accounting["failed_attempts"] == 1
    assert accounting["ok"] is True


def test_counts_reconcile_rejects_orphan_and_duplicate_exclusions():
    ledger = Ledger()
    ledger.add_exclusion(ExclusionRecord(
        candidate_id="ghost", reason=ExclusionReason.FETCH_FAILED))
    accounting = ledger.accounting()
    assert accounting["orphan_exclusion_candidate_ids"] == ["ghost"]
    assert accounting["ok"] is False

    ledger2 = Ledger()
    ledger2.register_candidate(_ledger_candidate())
    ledger2.set_candidate_state("c1", CandidateState.EXCLUDED)
    for _ in range(2):
        ledger2.add_exclusion(ExclusionRecord(
            candidate_id="c1", reason=ExclusionReason.SCANNED_PDF))
    accounting2 = ledger2.accounting()
    assert accounting2["permanently_excluded"] == 1
    assert accounting2["duplicate_exclusion_records"] == 1
    assert accounting2["ok"] is False


def test_counts_reconcile_reports_missing_candidate_id():
    ledger = Ledger()
    accounting = ledger.accounting(["missing"])
    assert accounting["missing_candidate_ids"] == ["missing"]
    assert accounting["unresolved"] == 1
    assert accounting["ok"] is False


def _write_bulk_csv(tmp_path, url: str = "https://x/1.pdf") -> str:
    p = tmp_path / "candidates.csv"
    p.write_text(
        "company_id,document_type,fiscal_period,url\n"
        f"LT,annual,FY2024,{url}\n",
        encoding="utf-8",
    )
    return str(p)


def test_bulk_summary_passes_and_prints_accounting(tmp_path, capsys, monkeypatch):
    import src.ingestion.pipeline as pipeline

    def fake_run_bulk(candidates, fetcher, ledger, storage,
                      storage_key_prefix=""):
        for candidate in candidates:
            ledger.register_candidate(candidate)
            ledger.set_candidate_state(candidate.candidate_id, CandidateState.EXTRACTED)
        return {"extracted": len(candidates)}, []

    monkeypatch.setattr(pipeline, "run_bulk", fake_run_bulk)
    rc = pipeline.main([
        "bulk", "--candidates", _write_bulk_csv(tmp_path),
        "--ledger", str(tmp_path / "ledger.jsonl"),
        "--storage-dir", str(tmp_path / "raw"),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "accounting invariant: PASS" in out
    assert "accounting accepted: 1" in out
    assert "accounting by_state: {'extracted': 1}" in out


def test_bulk_summary_fails_nonzero_on_unresolved_gap(tmp_path, capsys, monkeypatch):
    import src.ingestion.pipeline as pipeline

    def fake_run_bulk(candidates, fetcher, ledger, storage,
                      storage_key_prefix=""):
        for candidate in candidates:
            ledger.register_candidate(candidate)
        return {"discovered": len(candidates)}, []

    monkeypatch.setattr(pipeline, "run_bulk", fake_run_bulk)
    rc = pipeline.main([
        "bulk", "--candidates", _write_bulk_csv(tmp_path),
        "--ledger", str(tmp_path / "ledger.jsonl"),
        "--storage-dir", str(tmp_path / "raw"),
    ])
    out = capsys.readouterr().out
    assert rc == 2
    assert "accounting invariant: FAIL" in out
    assert "accounting gap:" in out
    assert "unresolved=" in out


def test_bulk_summary_counts_duplicate_rows_separately(tmp_path, capsys, monkeypatch):
    import src.ingestion.pipeline as pipeline

    def fake_run_bulk(candidates, fetcher, ledger, storage,
                      storage_key_prefix=""):
        for candidate in candidates:
            ledger.register_candidate(candidate)
            ledger.set_candidate_state(candidate.candidate_id, CandidateState.EXTRACTED)
        return {"extracted": len(candidates), "duplicate": 1}, []

    monkeypatch.setattr(pipeline, "run_bulk", fake_run_bulk)
    rc = pipeline.main([
        "bulk", "--candidates", _write_bulk_csv(tmp_path),
        "--ledger", str(tmp_path / "ledger.jsonl"),
        "--storage-dir", str(tmp_path / "raw"),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "duplicate: 1" in out
    assert "accounting invariant: PASS" in out


def test_run_bulk_keeps_same_tuple_without_announcement_id(tmp_path):
    from src.ingestion.pipeline import run_bulk

    pdf1 = _make_pdf()
    pdf2 = _make_pdf() + b"\n"
    fetcher = _CountingFetcher({
        "https://ok/1.pdf": pdf1,
        "https://mirror/1.pdf": pdf2,
    })
    ledger = Ledger()
    storage = LocalRawStorage(tmp_path)
    first = _bulk_candidate("disc-a", "https://ok/1.pdf")
    possible_revision = _bulk_candidate("disc-b", "https://mirror/1.pdf")
    counts, quarantined = run_bulk(
        [first, possible_revision], fetcher=fetcher, ledger=ledger, storage=storage)
    assert fetcher.calls == ["https://ok/1.pdf", "https://mirror/1.pdf"]
    assert counts["duplicate"] == 0
    assert len(ledger.rows) == 2
    assert quarantined == []


def test_run_bulk_keeps_revision_with_different_announcement_id(tmp_path):
    from src.ingestion.pipeline import run_bulk

    pdf1 = _make_pdf()
    pdf2 = _make_pdf() + b"\n"
    fetcher = _CountingFetcher({
        "https://ok/1.pdf": pdf1,
        "https://mirror/1.pdf": pdf2,
    })
    ledger = Ledger()
    storage = LocalRawStorage(tmp_path)
    first = _bulk_candidate("disc-a", "https://ok/1.pdf", announcement_id="ANN-1")
    revision = _bulk_candidate("disc-b", "https://mirror/1.pdf", announcement_id="ANN-2")
    counts, quarantined = run_bulk(
        [first, revision], fetcher=fetcher, ledger=ledger, storage=storage)
    assert fetcher.calls == ["https://ok/1.pdf", "https://mirror/1.pdf"]
    assert counts["duplicate"] == 0
    assert len(ledger.rows) == 2
    assert quarantined == []
