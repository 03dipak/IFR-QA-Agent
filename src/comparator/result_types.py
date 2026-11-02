"""Comparator result types and matrix builder.

Implements lld_comparator.txt section 6, 7, 8: closed result vocabulary,
restatement handling, partial results under the step cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .gate import ComparabilityResult


class ResultType(StrEnum):
    COMPARABLE = "comparable"
    NOT_COMPARABLE = "not_comparable"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    NOT_EVALUATED = "not_evaluated"
    NOT_APPLICABLE = "not_applicable"


class ExecutionStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class EvidenceInsufficiencyReason(StrEnum):
    NO_DOCUMENT = "no_document"
    NO_RELEVANT_TABLE = "no_relevant_table"
    EXTRACTION_UNUSABLE = "extraction_unusable"
    VALUE_UNRESOLVED = "value_unresolved"
    UNIT_UNRESOLVED = "unit_unresolved"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    CRITIC_VERIFICATION_FAILED = "critic_verification_failed"


@dataclass
class CellResult:
    cell_id: str
    company_id: str
    fiscal_period_id: str
    metric_id: str
    result_type: ResultType
    reason_codes: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    displayed_value: str = ""
    execution_status: ExecutionStatus = ExecutionStatus.COMPLETED
    restated: bool = False
    originating_document_id: str | None = None
    comparability: ComparabilityResult | None = None


@dataclass
class ComparisonMatrix:
    comparison_id: str
    expected_observation_count: int
    completed_observation_count: int = 0
    execution_status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    cells: list[CellResult] = field(default_factory=list)
    comparison_policy_version: str = "v1"

    @property
    def completeness_tally(self) -> str:
        return f"{self.completed_observation_count}/{self.expected_observation_count}"

    def add_cell(self, cell: CellResult) -> None:
        self.cells.append(cell)
        if cell.execution_status == ExecutionStatus.COMPLETED:
            self.completed_observation_count += 1


@dataclass
class RestatementInfo:
    originating_document_id: str
    restated: bool
    reason: str = ""


def build_partial_matrix(comparison_id: str, expected: int,
                         cells: list[CellResult]) -> ComparisonMatrix:
    m = ComparisonMatrix(comparison_id=comparison_id,
                         expected_observation_count=expected)
    for c in cells:
        m.add_cell(c)
    m.execution_status = (ExecutionStatus.COMPLETED
                         if m.completed_observation_count == expected
                         else ExecutionStatus.IN_PROGRESS)
    return m


def cite_originating(cell: CellResult, originating_doc: str,
                     restated: bool) -> CellResult:
    cell.originating_document_id = originating_doc
    cell.restated = restated
    return cell
