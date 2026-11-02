"""Unit tests for LLD-11 comparability gate."""

from __future__ import annotations

from src.comparator import (
    AlignmentClass,
    AuditPreference,
    AuditStatus,
    Observation,
    ReportingBasis,
    TargetBasis,
    TemporalType,
    gate_cell,
)


def _obs(**kw):
    defaults = dict(
        observation_id="o1", company_id="C1", fiscal_period_id="FY25",
        metric_id="revenue", reporting_basis=ReportingBasis.CONSOLIDATED,
        audit_status=AuditStatus.AUDITED, currency="INR", dimension="money",
        period_alignment=AlignmentClass.EXACT_SPAN, period_duration="12M",
        temporal_type=TemporalType.FLOW,
    )
    defaults.update(kw)
    return Observation(**defaults)


def test_comparable_basic():
    obs = _obs()
    target = TargetBasis()
    r = gate_cell(obs, target)
    assert r.status == "COMPARABLE"


def test_currency_mismatch():
    obs = _obs(currency="USD")
    r = gate_cell(obs, TargetBasis())
    assert r.status == "NOT_COMPARABLE"
    assert "currency" in r.mismatched_dimensions


def test_dimension_mismatch():
    obs = _obs(dimension="per_share")
    r = gate_cell(obs, TargetBasis())
    assert r.status == "NOT_COMPARABLE"
    assert "dimension" in r.mismatched_dimensions


def test_reporting_basis_mismatch():
    obs = _obs(reporting_basis=ReportingBasis.STANDALONE)
    r = gate_cell(obs, TargetBasis())
    assert r.status == "NOT_COMPARABLE"
    assert "reporting_basis" in r.mismatched_dimensions


def test_audit_require_audited():
    obs = _obs(audit_status=AuditStatus.UNAUDITED)
    target = TargetBasis(audit_preference=AuditPreference.REQUIRE_AUDITED)
    r = gate_cell(obs, target)
    assert r.status == "NOT_COMPARABLE"
    assert "audit_status" in r.mismatched_dimensions


def test_audit_allow_with_disclosure():
    obs = _obs(audit_status=AuditStatus.UNAUDITED)
    target = TargetBasis(audit_preference=AuditPreference.ALLOW_WITH_DISCLOSURE)
    r = gate_cell(obs, target)
    assert r.status == "COMPARABLE"
    assert any("audit_disclosure" in q for q in r.qualifications)


def test_period_transition_refused():
    obs = _obs(period_alignment=AlignmentClass.TRANSITION_PERIOD)
    r = gate_cell(obs, TargetBasis())
    assert r.status == "NOT_COMPARABLE"
    assert "period_alignment" in r.mismatched_dimensions


def test_stock_point_in_time():
    obs = _obs(temporal_type=TemporalType.STOCK,
               period_alignment=AlignmentClass.POINT_IN_TIME,
               period_duration="0M")
    target = TargetBasis(temporal_type=TemporalType.STOCK, period_duration="0M")
    r = gate_cell(obs, target)
    assert r.status == "COMPARABLE"


def test_sector_not_applicable():
    r = gate_cell(_obs(), TargetBasis(), metric_applicable_formats=["manufacturing"],
                  company_format="bank")
    assert r.status == "NOT_APPLICABLE"


def test_same_duration_different_span_qualification():
    obs = _obs(period_alignment=AlignmentClass.SAME_DURATION_DIFFERENT_SPAN)
    r = gate_cell(obs, TargetBasis())
    assert r.status == "COMPARABLE"
    assert "same_duration_different_span" in r.qualifications
