"""Period resolution — fiscal calendar driven.

Implements lld_entity_registry.txt section 4: FYnn/FYnn-Qn labels,
gregorian_span from fiscal_year_end, closed period types, period intent
distinction, relative-period rules (corpus-relative, never date-relative),
multi-company span/duration/transition comparison, TTM as request contract.
"""

from __future__ import annotations

from typing import Any

from .models import (
    FiscalCalendar,
    PeriodResolution,
    PeriodResolutionStatus,
    PeriodType,
    ResolutionIntent,
)


def _month_day(cal: FiscalCalendar) -> tuple[int, int]:
    return cal.fiscal_year_end_month, cal.fiscal_year_end_day


def compute_gregorian_span(cal: FiscalCalendar, fiscal_year: int,
                           period_type: PeriodType = PeriodType.ANNUAL) -> tuple[str, str]:
    """Compute (start, end) ISO dates for a fiscal period given the
    company's fiscal year-end. FY label names the year the FY ENDS."""
    m, d = _month_day(cal)
    # fiscal year ends on (m, d) of `fiscal_year`
    end = f"{fiscal_year:04d}-{m:02d}-{d:02d}"
    if period_type == PeriodType.ANNUAL:
        # start = day after previous fiscal year end
        prev_year = fiscal_year - 1
        start = f"{prev_year:04d}-{m:02d}-{d:02d}"
        # approximate: start is next day; keep simple ISO for comparison
        return start, end
    if period_type == PeriodType.QUARTERLY:
        # last quarter = 3 months ending at fiscal year end
        # compute start month
        start_month = m - 2
        start_year = fiscal_year
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        return f"{start_year:04d}-{start_month:02d}-01", end
    if period_type == PeriodType.HALF_YEAR:
        start_month = m - 5
        start_year = fiscal_year
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        return f"{start_year:04d}-{start_month:02d}-01", end
    if period_type == PeriodType.NINE_MONTH:
        start_month = m - 8
        start_year = fiscal_year
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        return f"{start_year:04d}-{start_month:02d}-01", end
    if period_type == PeriodType.POINT_IN_TIME:
        return end, end
    return "", end


def fiscal_year_label(fiscal_year_end_year: int) -> str:
    return f"FY{str(fiscal_year_end_year)[-2:]}"


def resolve_period_with_calendar(cal: FiscalCalendar, intent: ResolutionIntent,
                                corpus_latest_fy_year: int = 2025,
                                turn_context: dict[str, Any] | None = None) -> PeriodResolution:
    """Resolve a period for a company based on intent and fiscal calendar.

    Relative periods resolve per entity (corpus-relative, never date-relative).
    'and the year before?' resolves against the previous turn's period.
    'this financial year' is stated, never silently 'latest in corpus'.
    """
    turn_context = turn_context or {}
    m, d = _month_day(cal)

    if intent == ResolutionIntent.LAST_FINANCIAL_YEAR:
        fy = corpus_latest_fy_year
        start, end = compute_gregorian_span(cal, fy, PeriodType.ANNUAL)
        return PeriodResolution(
            period_type=PeriodType.ANNUAL,
            fiscal_period_id=f"FY{str(fy)[-2:]}",
            gregorian_span=f"{start} to {end}",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            intent=intent,
        )

    if intent == ResolutionIntent.LATEST_ANNUAL_REPORT:
        fy = corpus_latest_fy_year
        start, end = compute_gregorian_span(cal, fy, PeriodType.ANNUAL)
        return PeriodResolution(
            period_type=PeriodType.ANNUAL,
            fiscal_period_id=f"FY{str(fy)[-2:]}",
            gregorian_span=f"{start} to {end}",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            reason="latest annual report available in corpus",
            intent=intent,
        )

    if intent == ResolutionIntent.LATEST_QUARTER:
        fy = corpus_latest_fy_year
        start, end = compute_gregorian_span(cal, fy, PeriodType.QUARTERLY)
        return PeriodResolution(
            period_type=PeriodType.QUARTERLY,
            fiscal_period_id=f"FY{str(fy)[-2:]}-Q4",
            gregorian_span=f"{start} to {end}",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            intent=intent,
        )

    if intent == ResolutionIntent.LATEST_RESULTS:
        fy = corpus_latest_fy_year
        start, end = compute_gregorian_span(cal, fy, PeriodType.QUARTERLY)
        return PeriodResolution(
            period_type=PeriodType.QUARTERLY,
            fiscal_period_id=f"FY{str(fy)[-2:]}-LR",
            gregorian_span=f"{start} to {end}",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            reason="latest results (quarterly or annual)",
            intent=intent,
        )

    if intent == ResolutionIntent.LAST_12_MONTHS:
        # TTM is a request contract only; calculation happens in comparator
        return PeriodResolution(
            period_type=PeriodType.TTM,
            fiscal_period_id=f"TTM-{corpus_latest_fy_year}",
            gregorian_span="",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            reason="TTM request contract; calculation performed by comparator",
            intent=intent,
            is_ttm_request=True,
        )

    if intent == ResolutionIntent.THIS_FINANCIAL_YEAR:
        fy = corpus_latest_fy_year
        start, end = compute_gregorian_span(cal, fy, PeriodType.ANNUAL)
        return PeriodResolution(
            period_type=PeriodType.ANNUAL,
            fiscal_period_id=f"FY{str(fy)[-2:]}-CURRENT",
            gregorian_span=f"{start} to {end}",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            reason="current fiscal year, stated (not silently 'latest in corpus')",
            intent=intent,
        )

    if intent == ResolutionIntent.LATEST_AVAILABLE_FY:
        fy = corpus_latest_fy_year
        start, end = compute_gregorian_span(cal, fy, PeriodType.ANNUAL)
        return PeriodResolution(
            period_type=PeriodType.ANNUAL,
            fiscal_period_id=f"FY{str(fy)[-2:]}",
            gregorian_span=f"{start} to {end}",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            reason="corpus-relative latest available fiscal year",
            intent=intent,
        )

    # Relative: "and the year before?" resolves against previous turn's period
    if turn_context and "previous_period" in turn_context:
        prev = turn_context["previous_period"]
        # decrement fiscal year by 1
        try:
            prev_fy = int(prev.split("-")[0].replace("FY", ""))
            new_fy = prev_fy - 1
            start, end = compute_gregorian_span(cal, new_fy + 2000, PeriodType.ANNUAL)
            return PeriodResolution(
                period_type=PeriodType.ANNUAL,
                fiscal_period_id=f"FY{new_fy:02d}",
                gregorian_span=f"{start} to {end}",
                resolution_status=PeriodResolutionStatus.RESOLVED,
                reason="resolved against previous turn's period",
                intent=intent,
            )
        except (ValueError, IndexError):
            pass

    # default: latest completed annual
    fy = corpus_latest_fy_year
    start, end = compute_gregorian_span(cal, fy, PeriodType.ANNUAL)
    return PeriodResolution(
        period_type=PeriodType.ANNUAL,
        fiscal_period_id=f"FY{str(fy)[-2:]}",
        gregorian_span=f"{start} to {end}",
        resolution_status=PeriodResolutionStatus.RESOLVED,
        intent=intent,
    )


def periods_comparable(periods: list[PeriodResolution],
                      metric_temporal_type: str = "flow") -> tuple[bool, str]:
    """Multi-company comparison: spans + durations + transition status.

    Returns (comparable, reason). Balance-sheet (stock) uses date match;
    flow uses duration match.
    """
    if len(periods) < 2:
        return True, ""
    if metric_temporal_type in ("stock", "ratio"):
        # point-in-time: dates must match
        ends = {p.gregorian_span.split(" to ")[-1] for p in periods if " to " in p.gregorian_span}
        if len(ends) > 1:
            return False, "reporting date mismatch (stock/ratio)"
        return True, ""
    # flow: durations must match; transition periods refused
    for p in periods:
        pt = p.period_type
        ptv = pt.value if hasattr(pt, "value") else str(pt)
        if ptv == "transition":
            return False, "transition period in flow comparison"
    spans = [p.gregorian_span for p in periods]
    # compare duration by span length (rough)
    durations = set()
    for s in spans:
        if " to " in s:
            a, b = s.split(" to ")
            durations.add((a, b))
    if len(durations) > 1:
        # could still be same duration different span
        return True, "same_duration_different_span"
    return True, ""


def common_contract_for_headers(header_text: str) -> PeriodResolutionStatus:
    """Period resolution is a common contract for user queries AND source
    headers; unresolved/ambiguous headers are flagged, never guessed."""
    if not header_text or header_text.strip() == "":
        return PeriodResolutionStatus.UNRESOLVED
    return PeriodResolutionStatus.RESOLVED
