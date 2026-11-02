"""Comparator data models.

Implements lld_comparator.txt sections 4-5: ComparisonPlan, metric definitions,
comparability policies, matrix schema (logical).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TemporalType(StrEnum):
    FLOW = "flow"
    STOCK = "stock"
    RATIO = "ratio"
    DERIVED = "derived"


class MetricSourceType(StrEnum):
    REPORTED = "reported"
    DERIVED = "derived"
    EITHER_WITH_POLICY = "either_with_policy"


class ReportingBasis(StrEnum):
    CONSOLIDATED = "consolidated"
    STANDALONE = "standalone"
    PARENT_ONLY = "parent_only"
    UNCLEAR = "unclear"


class AuditPreference(StrEnum):
    REQUIRE_SAME_STATUS = "require_same_status"
    ALLOW_WITH_DISCLOSURE = "allow_with_disclosure"
    REQUIRE_AUDITED = "require_audited"
    NOT_APPLICABLE = "not_applicable"


class ValueSelection(StrEnum):
    ORIGINALLY_REPORTED = "originally_reported"
    LATEST_RESTATED = "latest_restated"
    LATEST_AVAILABLE = "latest_available"


class AuditStatus(StrEnum):
    AUDITED = "audited"
    REVIEWED = "reviewed"
    UNAUDITED = "unaudited"
    UNREVIEWED = "unreviewed"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class AlignmentClass(StrEnum):
    EXACT_SPAN = "exact_span"
    SAME_DURATION_DIFFERENT_SPAN = "same_duration_different_span"
    OVERLAPPING_SPAN = "overlapping_span"
    NON_OVERLAPPING_SPAN = "non_overlapping_span"
    TRANSITION_PERIOD = "transition_period"
    POINT_IN_TIME = "point_in_time"
    UNKNOWN = "unknown"


@dataclass
class MetricDefinition:
    metric_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    statement_kind: str = ""
    temporal_type: TemporalType = TemporalType.FLOW
    measurement_dimension: str = "money"
    measurement_unit: str = "INR"
    formula_id: str | None = None
    metric_source_type: MetricSourceType = MetricSourceType.REPORTED
    reporting_basis: ReportingBasis = ReportingBasis.CONSOLIDATED
    inclusion_rules: list[str] = field(default_factory=list)
    exclusion_rules: list[str] = field(default_factory=list)
    comparability_group: str = ""
    applicable_statement_formats: list[str] = field(default_factory=list)
    metric_definition_version: str = "v1"


@dataclass
class ComparisonPlan:
    companies: list[str]  # resolved entity ids
    periods: list[str]    # registry-resolved period labels
    metrics: list[str]    # metric ids
    reporting_basis: ReportingBasis = ReportingBasis.CONSOLIDATED
    audit_preference: AuditPreference = AuditPreference.ALLOW_WITH_DISCLOSURE
    period_duration: str = "12M"
    value_selection: ValueSelection = ValueSelection.LATEST_AVAILABLE
    ordering_request: str | None = None  # one factual metric id


@dataclass
class PlanValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Evaluative words forbidden in ordering_request
EVALUATIVE_WORDS = {"best", "strongest", "weakest", "rank", "top", "winner",
                    "leading", "outperform", "superior", "highest", "lowest"}


def validate_plan(plan: ComparisonPlan,
                  registry: dict[str, Any] | None = None,
                  metric_defs: dict[str, MetricDefinition] | None = None) -> PlanValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    registry = registry or {}
    metric_defs = metric_defs or {}

    # company count
    if not (1 <= len(plan.companies) <= 3):
        errors.append(f"companies must be 1-3, got {len(plan.companies)}")
    # resolved entity ids only
    for c in plan.companies:
        if registry and c not in registry:
            errors.append(f"unresolved company id: {c}")
    if len(set(plan.companies)) != len(plan.companies):
        errors.append("duplicate company ids")

    # periods
    if not (1 <= len(plan.periods) <= 3):
        errors.append(f"periods must be 1-3, got {len(plan.periods)}")
    # periods must be registry-resolved (non-empty strings)
    for p in plan.periods:
        if not p or not isinstance(p, str):
            errors.append(f"invalid period: {p!r}")

    # metrics
    if not (1 <= len(plan.metrics) <= 4):
        errors.append(f"metrics must be 1-4, got {len(plan.metrics)}")
    for m in plan.metrics:
        if metric_defs and m not in metric_defs:
            errors.append(f"unknown metric id: {m}")
    if len(set(plan.metrics)) != len(plan.metrics):
        errors.append("duplicate metric ids")

    # matrix cap: 3x3x4 = 36 observation slots
    slots = len(plan.companies) * len(plan.periods) * len(plan.metrics)
    if slots > 36:
        errors.append(f"matrix cap exceeded: {slots} observation slots > 36")

    # single company + single period is not a comparison
    if len(plan.companies) == 1 and len(plan.periods) == 1:
        warnings.append("single company/single period is not a comparison; consider plain QA")

    # ordering_request: one factual metric, no evaluative words
    if plan.ordering_request is not None:
        if plan.ordering_request not in plan.metrics:
            errors.append("ordering_request must be one of the plan metrics")
        low = plan.ordering_request.lower()
        for w in EVALUATIVE_WORDS:
            if w in low.split():
                errors.append(f"evaluative word in ordering_request: {w}")

    # reporting_basis valid enum
    if plan.reporting_basis not in ReportingBasis:
        errors.append(f"invalid reporting_basis: {plan.reporting_basis}")

    # audit_preference valid
    if plan.audit_preference not in AuditPreference:
        errors.append(f"invalid audit_preference: {plan.audit_preference}")

    # value_selection valid
    if plan.value_selection not in ValueSelection:
        errors.append(f"invalid value_selection: {plan.value_selection}")

    return PlanValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
