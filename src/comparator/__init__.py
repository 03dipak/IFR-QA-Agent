"""Comparator + mode-aware comparability gate.

Implements lld_comparator.txt — Tool 3: ComparisonPlan JSON schema,
comparability gate, partial results, restatement handling.
"""

from .gate import (
    ComparabilityResult,
    Observation,
    TargetBasis,
    gate_cell,
)
from .models import (
    AlignmentClass,
    AuditPreference,
    AuditStatus,
    ComparisonPlan,
    MetricDefinition,
    MetricSourceType,
    PlanValidationResult,
    ReportingBasis,
    TemporalType,
    ValueSelection,
    validate_plan,
)
from .result_types import (
    CellResult,
    ComparisonMatrix,
    EvidenceInsufficiencyReason,
    ExecutionStatus,
    ResultType,
    build_partial_matrix,
    cite_originating,
)

__all__ = [
    "AuditPreference", "AuditStatus", "AlignmentClass", "ComparisonPlan",
    "CellResult", "ComparabilityResult", "ComparisonMatrix", "EvidenceInsufficiencyReason",
    "ExecutionStatus", "MetricDefinition", "MetricSourceType", "Observation",
    "PlanValidationResult", "ReportingBasis", "ResultType", "TargetBasis",
    "TemporalType", "ValueSelection", "build_partial_matrix", "cite_originating",
    "gate_cell", "validate_plan",
]
