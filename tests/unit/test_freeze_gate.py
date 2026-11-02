"""Unit tests for LLD-03 freeze gate."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.ingestion.freeze_manager import (
    CandidateAccounting,
    FreezeManager,
    GoldEvidenceRef,
)
from src.ingestion.models import (
    AuditStatus,
    DocType,
    DocumentLedgerRow,
    ExtractionStatus,
    PageExtract,
)


def _row() -> DocumentLedgerRow:
    return DocumentLedgerRow(
        document_id="d1", filing_id="f1", doc_type=DocType.ANNUAL,
        company_id="C1", fiscal_year="FY25", reporting_period="FY",
        publication_date=datetime.now(UTC), page_count=1,
        checksum="abc", extraction_status=ExtractionStatus.EXTRACTED,
        extraction_config_hash="cfg", audit_status=AuditStatus.UNCLEAR,
    )


def _pe() -> PageExtract:
    return PageExtract(document_id="d1", extraction_run_id="r1",
                      pdf_page_index=1, page_text="hi", extraction_hash="h")


def test_validate_freeze_passes():
    fm = FreezeManager()
    pe = _pe()
    m = fm.create_freeze("v1", None, [_row()], [pe], "cfg", "run1", "chash",
                         qa_report_checksum="q", qa_passed=True)
    res = fm.validate_freeze(m.freeze_id, CandidateAccounting(1, 1, 0, 0), [], [pe])
    assert res.passed is True
    assert m.freeze_gate_passed is True


def test_validate_freeze_unresolved_fails():
    fm = FreezeManager()
    pe = _pe()
    m = fm.create_freeze("v1", None, [_row()], [pe], "cfg", "run1", "chash",
                         qa_report_checksum="q", qa_passed=True)
    res = fm.validate_freeze(m.freeze_id, CandidateAccounting(2, 1, 0, 1), [], [pe])
    assert res.passed is False
    assert any("unresolved" in r for r in res.reasons)


def test_validate_freeze_accounting_mismatch():
    fm = FreezeManager()
    pe = _pe()
    m = fm.create_freeze("v1", None, [_row()], [pe], "cfg", "run1", "chash",
                         qa_report_checksum="q", qa_passed=True)
    res = fm.validate_freeze(m.freeze_id, CandidateAccounting(5, 1, 0, 0), [], [pe])
    assert res.passed is False


def test_validate_freeze_gold_ref_page_missing():
    fm = FreezeManager()
    pe = _pe()
    m = fm.create_freeze("v1", None, [_row()], [pe], "cfg", "run1", "chash",
                         qa_report_checksum="q", qa_passed=True)
    ref = GoldEvidenceRef(question_id="q1", document_id="d1", pdf_page_index=99, extraction_hash="h")
    res = fm.validate_freeze(m.freeze_id, CandidateAccounting(1, 1, 0, 0), [ref], [pe])
    assert res.passed is False
    assert any("page not in frozen set" in r for r in res.reasons)


def test_validate_freeze_gold_ref_hash_mismatch():
    fm = FreezeManager()
    pe = _pe()
    m = fm.create_freeze("v1", None, [_row()], [pe], "cfg", "run1", "chash",
                         qa_report_checksum="q", qa_passed=True)
    ref = GoldEvidenceRef(question_id="q1", document_id="d1", pdf_page_index=1, extraction_hash="wrong")
    res = fm.validate_freeze(m.freeze_id, CandidateAccounting(1, 1, 0, 0), [ref], [pe])
    assert res.passed is False
    assert any("extraction hash mismatch" in r for r in res.reasons)


def test_publish_requires_freeze_gate():
    fm = FreezeManager()
    pe = _pe()
    m = fm.create_freeze("v1", None, [_row()], [pe], "cfg", "run1", "chash",
                         qa_report_checksum="q", qa_passed=True)
    m.manifest_checksum = m.compute_manifest_checksum()
    with pytest.raises(ValueError):
        fm.publish(m.freeze_id)


def test_load_verified():
    fm = FreezeManager()
    pe = _pe()
    m = fm.create_freeze("v1", None, [_row()], [pe], "cfg", "run1", "chash",
                         qa_report_checksum="q", qa_passed=True)
    fm.validate_freeze(m.freeze_id, CandidateAccounting(1, 1, 0, 0), [], [pe])
    fm.publish(m.freeze_id)
    loaded = fm.load_verified(m.freeze_id)
    assert loaded.freeze_id == m.freeze_id


def test_reconcile():
    acc = CandidateAccounting(discovered=10, accepted=7, permanently_excluded=3, unresolved=0)
    assert acc.reconciles() is True
    acc2 = CandidateAccounting(discovered=10, accepted=7, permanently_excluded=2, unresolved=0)
    assert acc2.reconciles() is False
