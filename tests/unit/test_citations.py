"""Unit tests for LLD-15 citations."""

from __future__ import annotations

from src.citations import (
    Claim,
    Context,
    EvidenceObject,
    EvidenceType,
    Figure,
    HandleTable,
    PageLabelStatus,
    PageRef,
    VerificationOutcome,
    compute_citation_coverage,
    compute_citation_precision,
    derived_verify,
    direct_verify,
)


def test_evidence_id_deterministic():
    a = EvidenceObject.make_evidence_id("d1", 5, "100", "t1:r1:c1")
    b = EvidenceObject.make_evidence_id("d1", 5, "100", "t1:r1:c1")
    assert a == b
    c = EvidenceObject.make_evidence_id("d1", 5, "100", "t1:r2:c1")
    assert a != c


def test_render_citation_dual():
    ev = EvidenceObject(
        evidence_id="e1", evidence_type=EvidenceType.TABLE_CELL, corpus_freeze_id="f1",
        document_id="d1", page=PageRef(pdf_page_index=214, printed_page_label="203",
        page_label_status=PageLabelStatus.DETECTED),
    )
    s = ev.render_citation(company="Acme", report="Annual Report FY25", filed_date="2025-06-30")
    assert "PDF p. 214 (printed p. 203)" in s
    assert "Acme" in s


def test_render_citation_equal():
    ev = EvidenceObject(
        evidence_id="e2", evidence_type=EvidenceType.PROSE, corpus_freeze_id="f1",
        document_id="d1", page=PageRef(pdf_page_index=10, printed_page_label="10",
        page_label_status=PageLabelStatus.DETECTED),
    )
    s = ev.render_citation()
    assert "PDF p. 10" in s
    assert "printed" not in s


def test_handle_table():
    ht = HandleTable(turn_id="t1")
    h = ht.mint("e1")
    assert h == 1
    assert ht.resolve(1) == "e1"
    assert ht.validate(1, "t1") is True
    assert ht.validate(1, "t2") is False
    assert ht.validate(99, "t1") is False


def test_direct_verify_source_not_found():
    ev = EvidenceObject(
        evidence_id="e", evidence_type=EvidenceType.PROSE, corpus_freeze_id="f",
        document_id="d", page=PageRef(pdf_page_index=1),
        figure=Figure(metric_id="revenue", raw="500"),
    )
    out = direct_verify(ev, "no such number here", metric="revenue", period="", basis="", unit="")
    assert out == VerificationOutcome.SOURCE_NOT_FOUND


def test_direct_verify_verified():
    ev = EvidenceObject(
        evidence_id="e", evidence_type=EvidenceType.PROSE, corpus_freeze_id="f",
        document_id="d", page=PageRef(pdf_page_index=1),
        figure=Figure(metric_id="revenue", raw="500"),
    )
    out = direct_verify(ev, "revenue was 500 crore", metric="revenue", period="", basis="", unit="")
    assert out == VerificationOutcome.VERIFIED


def test_direct_verify_substring_boundary():
    ev = EvidenceObject(
        evidence_id="e", evidence_type=EvidenceType.PROSE, corpus_freeze_id="f",
        document_id="d", page=PageRef(pdf_page_index=1),
        figure=Figure(metric_id="revenue", raw="1,234"),
    )
    out = direct_verify(ev, "total 11,234", metric="revenue", period="", basis="", unit="")
    assert out == VerificationOutcome.SOURCE_NOT_FOUND


def test_derived_verify_operand_mismatch():
    ev1 = EvidenceObject(evidence_id="a", evidence_type=EvidenceType.TABLE_CELL,
                          corpus_freeze_id="f", document_id="d",
                          page=PageRef(pdf_page_index=1),
                          context=Context(reporting_basis="consolidated"))
    ev2 = EvidenceObject(evidence_id="b", evidence_type=EvidenceType.TABLE_CELL,
                          corpus_freeze_id="f", document_id="d",
                          page=PageRef(pdf_page_index=1),
                          context=Context(reporting_basis="standalone"))
    from src.citations.models import DerivedEvidence
    de = DerivedEvidence(derived_evidence_id="de", formula_id="f",
                         formula_version="v1", calculation_result=10)
    out = derived_verify(de, [ev1, ev2], replay_result=10)
    assert out == VerificationOutcome.SEMANTIC_MISMATCH


def test_derived_verify_calc_mismatch():
    ev1 = EvidenceObject(evidence_id="a", evidence_type=EvidenceType.TABLE_CELL,
                          corpus_freeze_id="f", document_id="d",
                          page=PageRef(pdf_page_index=1),
                          context=Context(reporting_basis="consolidated"))
    from src.citations.models import DerivedEvidence
    de = DerivedEvidence(derived_evidence_id="de", formula_id="f",
                         formula_version="v1", calculation_result=10)
    out = derived_verify(de, [ev1], replay_result=99)
    assert out == VerificationOutcome.CALCULATION_MISMATCH


def test_citation_metrics():
    assert compute_citation_precision(3, 4) == 0.75
    assert compute_citation_coverage(2, 4) == 0.5
    assert compute_citation_precision(0, 0) == 1.0


def test_claim_materiality_default():
    c = Claim(claim_id="c1", claim_text="revenue 500")
    assert c.materiality == "material"
