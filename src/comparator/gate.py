"""Mode-aware comparability gate.

Implements lld_comparator.txt section 6: metric-aware gate, period-alignment
classes, audit policy, dimensional compatibility, restatement consumption.
Each cell is gated against a target basis, not against other cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    AlignmentClass,
    AuditPreference,
    AuditStatus,
    ReportingBasis,
    TemporalType,
)


@dataclass
class Observation:
    """One company's value for a period+metric (section 10)."""
    observation_id: str
    company_id: str
    fiscal_period_id: str
    metric_id: str
    reporting_basis: ReportingBasis
    audit_status: AuditStatus
    currency: str
    dimension: str  # money | per_share | percent | count | ratio
    period_alignment: AlignmentClass
    period_duration: str
    temporal_type: TemporalType
    normalized_value_ref: str = ""
    displayed_value: str = ""
    restatement_status: str | None = None


@dataclass
class TargetBasis:
    """Target basis the gate checks each cell against."""
    reporting_basis: ReportingBasis = ReportingBasis.CONSOLIDATED
    audit_preference: AuditPreference = AuditPreference.ALLOW_WITH_DISCLOSURE
    period_duration: str = "12M"
    currency: str = "INR"
    dimension: str = "money"
    temporal_type: TemporalType = TemporalType.FLOW


@dataclass
class ComparabilityResult:
    status: str  # COMPARABLE | NOT_COMPARABLE | NOT_APPLICABLE
    matched_dimensions: list[str] = field(default_factory=list)
    mismatched_dimensions: list[str] = field(default_factory=list)
    qualifications: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    comparison_policy_version: str = "v1"


def _audit_ok(obs: Observation, target: TargetBasis) -> tuple[bool, list[str]]:
    pref = target.audit_preference
    if pref == AuditPreference.NOT_APPLICABLE:
        return True, []
    if pref == AuditPreference.REQUIRE_AUDITED:
        if obs.audit_status != AuditStatus.AUDITED:
            return False, ["audit_status"]
        return True, []
    if pref == AuditPreference.REQUIRE_SAME_STATUS:
        # same status is checked by caller comparing to target; here target has no status,
        # so treat as: audited/reviewed acceptable, others refused
        if obs.audit_status in (AuditStatus.AUDITED, AuditStatus.REVIEWED):
            return True, []
        return False, ["audit_status"]
    # ALLOW_WITH_DISCLOSURE
    if obs.audit_status in (AuditStatus.AUDITED, AuditStatus.REVIEWED):
        return True, []
    return True, [f"audit_disclosure:{obs.audit_status.value}"]


def _period_ok(obs: Observation, target: TargetBasis) -> tuple[bool, list[str]]:
    if obs.temporal_type == TemporalType.STOCK or obs.temporal_type == TemporalType.RATIO:
        # point-in-time: duration not the criterion; alignment must be exact or overlapping
        if obs.period_alignment in (AlignmentClass.POINT_IN_TIME, AlignmentClass.EXACT_SPAN,
                                    AlignmentClass.OVERLAPPING_SPAN):
            return True, []
        return False, ["period_alignment"]
    # flow
    if obs.period_alignment == AlignmentClass.TRANSITION_PERIOD:
        return False, ["transition_period"]
    if obs.period_duration != target.period_duration and target.period_duration != "as_printed":
        if obs.period_alignment == AlignmentClass.SAME_DURATION_DIFFERENT_SPAN:
            return True, ["same_duration_different_span"]
        return False, ["period_duration"]
    ok_aligns = (AlignmentClass.EXACT_SPAN, AlignmentClass.SAME_DURATION_DIFFERENT_SPAN,
                 AlignmentClass.OVERLAPPING_SPAN)
    if obs.period_alignment in ok_aligns:
        qs = []
        if obs.period_alignment == AlignmentClass.SAME_DURATION_DIFFERENT_SPAN:
            qs.append("same_duration_different_span")
        return True, qs
    return False, ["period_alignment"]


def gate_cell(obs: Observation, target: TargetBasis,
             metric_applicable_formats: list[str] | None = None,
             company_format: str | None = None) -> ComparabilityResult:
    matched: list[str] = []
    mismatched: list[str] = []
    qualifications: list[str] = []
    reasons: list[str] = []

    # sector applicability
    if metric_applicable_formats and company_format:
        if company_format not in metric_applicable_formats:
            return ComparabilityResult(
                status="NOT_APPLICABLE",
                mismatched_dimensions=["sector_applicability"],
                reason_codes=["not_applicable_format"],
            )

    # currency
    if obs.currency == target.currency:
        matched.append("currency")
    else:
        mismatched.append("currency")
        reasons.append("currency")

    # dimension
    if obs.dimension == target.dimension:
        matched.append("dimension")
    else:
        mismatched.append("dimension")
        reasons.append("dimension")

    # reporting basis
    if obs.reporting_basis == target.reporting_basis:
        matched.append("reporting_basis")
    else:
        mismatched.append("reporting_basis")
        reasons.append("reporting_basis")

    # audit
    audit_ok, audit_qs = _audit_ok(obs, target)
    if audit_ok:
        matched.append("audit_status")
        qualifications.extend(audit_qs)
    else:
        mismatched.append("audit_status")
        reasons.append("audit_status")

    # period
    period_ok, period_qs = _period_ok(obs, target)
    if period_ok:
        matched.append("period_alignment")
        qualifications.extend(period_qs)
    else:
        mismatched.append("period_alignment")
        reasons.append("period_alignment")

    if mismatched:
        return ComparabilityResult(
            status="NOT_COMPARABLE",
            matched_dimensions=matched,
            mismatched_dimensions=mismatched,
            qualifications=qualifications,
            reason_codes=reasons,
        )
    return ComparabilityResult(
        status="COMPARABLE",
        matched_dimensions=matched,
        mismatched_dimensions=[],
        qualifications=qualifications,
        reason_codes=[],
    )
