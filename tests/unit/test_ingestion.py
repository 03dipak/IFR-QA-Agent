"""Unit tests for LLD-01 ingestion pipeline."""

from __future__ import annotations

import hashlib
from datetime import UTC

import pytest

from src.ingestion.checksummer import sha256_bytes
from src.ingestion.dedupe import classify_mirror, filing_id
from src.ingestion.extractor import text_density_ok
from src.ingestion.freeze_manager import FreezeManager
from src.ingestion.ledger import Ledger
from src.ingestion.models import (
    Candidate,
    CandidateState,
    DocType,
    DocumentLedgerRow,
)
from src.ingestion.storage import LocalRawStorage


def _make_pdf(text: str = "Hello world revenue 100") -> bytes:
    # Minimal valid PDF with text
    from io import BytesIO

    from pypdf import PdfWriter
    w = PdfWriter()
    page = w.add_blank_page(width=200, height=200)
    # add text via annotation not needed; use a simple content stream
    w.add_page(page)
    buf = BytesIO()
    w.write(buf)
    return buf.getvalue()


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
