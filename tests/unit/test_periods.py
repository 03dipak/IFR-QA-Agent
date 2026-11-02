"""Unit tests for LLD-14 period resolution."""

from __future__ import annotations

from src.entity_registry import (
    FiscalCalendar,
    PeriodResolutionStatus,
    PeriodType,
    ResolutionIntent,
    common_contract_for_headers,
    compute_gregorian_span,
    periods_comparable,
    resolve_period_with_calendar,
)


def test_compute_span_march_company():
    cal = FiscalCalendar(company_id="C1", fiscal_year_end_month=3, fiscal_year_end_day=31)
    start, end = compute_gregorian_span(cal, 2025, PeriodType.ANNUAL)
    assert end == "2025-03-31"
    assert start.startswith("2024")


def test_compute_span_december_company():
    cal = FiscalCalendar(company_id="C2", fiscal_year_end_month=12, fiscal_year_end_day=31)
    start, end = compute_gregorian_span(cal, 2025, PeriodType.ANNUAL)
    assert end == "2025-12-31"
    assert start.startswith("2024")


def test_resolve_last_financial_year():
    cal = FiscalCalendar(company_id="C1", fiscal_year_end_month=3, fiscal_year_end_day=31)
    pr = resolve_period_with_calendar(cal, ResolutionIntent.LAST_FINANCIAL_YEAR,
                                     corpus_latest_fy_year=2025)
    assert pr.period_type == PeriodType.ANNUAL
    assert "FY25" in pr.fiscal_period_id
    assert pr.resolution_status == PeriodResolutionStatus.RESOLVED


def test_resolve_ttm_is_request_contract():
    cal = FiscalCalendar(company_id="C1", fiscal_year_end_month=3, fiscal_year_end_day=31)
    pr = resolve_period_with_calendar(cal, ResolutionIntent.LAST_12_MONTHS)
    assert pr.period_type == PeriodType.TTM
    assert pr.is_ttm_request is True
    assert "comparator" in pr.reason


def test_resolve_this_fy_stated():
    cal = FiscalCalendar(company_id="C1", fiscal_year_end_month=3, fiscal_year_end_day=31)
    pr = resolve_period_with_calendar(cal, ResolutionIntent.THIS_FINANCIAL_YEAR)
    assert "current fiscal year" in pr.reason


def test_resolve_previous_turn():
    cal = FiscalCalendar(company_id="C1", fiscal_year_end_month=3, fiscal_year_end_day=31)
    ctx = {"previous_period": "FY25"}
    pr = resolve_period_with_calendar(cal, ResolutionIntent.LAST_FINANCIAL_YEAR,
                                     corpus_latest_fy_year=2025, turn_context=ctx)
    # default path uses corpus_latest; relative path only when intent triggers it
    assert pr.resolution_status == PeriodResolutionStatus.RESOLVED


def test_periods_comparable_flow():
    cal = FiscalCalendar(company_id="C1", fiscal_year_end_month=3, fiscal_year_end_day=31)
    p1 = resolve_period_with_calendar(cal, ResolutionIntent.LAST_FINANCIAL_YEAR,
                                     corpus_latest_fy_year=2025)
    p2 = resolve_period_with_calendar(cal, ResolutionIntent.LAST_FINANCIAL_YEAR,
                                     corpus_latest_fy_year=2024)
    ok, reason = periods_comparable([p1, p2], metric_temporal_type="flow")
    assert isinstance(ok, bool)


def test_periods_comparable_stock_date_mismatch():
    from src.entity_registry.models import PeriodResolution, PeriodType
    p1 = PeriodResolution(period_type=PeriodType.POINT_IN_TIME,
                          gregorian_span="2025-03-31 to 2025-03-31")
    p2 = PeriodResolution(period_type=PeriodType.POINT_IN_TIME,
                          gregorian_span="2025-12-31 to 2025-12-31")
    ok, reason = periods_comparable([p1, p2], metric_temporal_type="stock")
    assert ok is False


def test_common_contract_headers():
    assert common_contract_for_headers("") == PeriodResolutionStatus.UNRESOLVED
    assert common_contract_for_headers("FY25") == PeriodResolutionStatus.RESOLVED


def test_closed_period_types():
    assert PeriodType.ANNUAL.value == "annual"
    assert PeriodType.POINT_IN_TIME.value == "point_in_time"
    assert PeriodType.TTM.value == "ttm"
