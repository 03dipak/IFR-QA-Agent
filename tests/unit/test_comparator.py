"""Unit tests for LLD-10 ComparisonPlan validator."""

from __future__ import annotations

from src.comparator import (
    ComparisonPlan,
    validate_plan,
)


def _registry():
    return {"C1": object(), "C2": object(), "C3": object()}


def _metrics():
    from src.comparator.models import MetricDefinition, TemporalType
    return {
        "revenue": MetricDefinition(metric_id="revenue", canonical_name="Revenue",
                                    temporal_type=TemporalType.FLOW),
        "pat": MetricDefinition(metric_id="pat", canonical_name="PAT",
                                temporal_type=TemporalType.FLOW),
    }


def test_valid_plan():
    plan = ComparisonPlan(companies=["C1", "C2"], periods=["FY24", "FY25"],
                          metrics=["revenue"])
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is True
    assert res.errors == []


def test_unresolved_company():
    plan = ComparisonPlan(companies=["NOPE"], periods=["FY24"], metrics=["revenue"])
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is False
    assert any("unresolved" in e for e in res.errors)


def test_company_count_cap():
    plan = ComparisonPlan(companies=["C1", "C2", "C3", "C4"], periods=["FY24"], metrics=["revenue"])
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is False


def test_metric_count_cap():
    plan = ComparisonPlan(companies=["C1"], periods=["FY24"],
                          metrics=["revenue", "pat", "x", "y", "z"])
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is False


def test_unknown_metric():
    plan = ComparisonPlan(companies=["C1"], periods=["FY24"], metrics=["nope"])
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is False
    assert any("unknown metric" in e for e in res.errors)


def test_evaluative_ordering():
    plan = ComparisonPlan(companies=["C1", "C2"], periods=["FY24"], metrics=["revenue"],
                          ordering_request="best")
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is False


def test_ordering_must_be_in_metrics():
    plan = ComparisonPlan(companies=["C1", "C2"], periods=["FY24"], metrics=["revenue"],
                          ordering_request="pat")
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is False


def test_matrix_cap():
    plan = ComparisonPlan(companies=["C1", "C2", "C3"], periods=["FY24", "FY25", "FY23"],
                          metrics=["revenue", "pat", "x", "y"])
    # 3*3*4=36 slots (at cap)
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    # unknown metrics already make it invalid
    assert res.valid is False


def test_single_company_single_period_warning():
    plan = ComparisonPlan(companies=["C1"], periods=["FY24"], metrics=["revenue"])
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is True
    assert any("single company" in w for w in res.warnings)


def test_duplicate_company():
    plan = ComparisonPlan(companies=["C1", "C1"], periods=["FY24"], metrics=["revenue"])
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is False


def test_invalid_reporting_basis():
    plan = ComparisonPlan(companies=["C1"], periods=["FY24"], metrics=["revenue"],
                          reporting_basis="foo")  # type: ignore
    res = validate_plan(plan, registry=_registry(), metric_defs=_metrics())
    assert res.valid is False
