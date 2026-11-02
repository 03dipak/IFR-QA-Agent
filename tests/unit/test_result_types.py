"""Unit tests for LLD-12 result types."""

from __future__ import annotations

from src.comparator import (
    CellResult,
    ComparisonMatrix,
    EvidenceInsufficiencyReason,
    ExecutionStatus,
    ResultType,
    build_partial_matrix,
    cite_originating,
)


def test_cell_result_defaults():
    c = CellResult(cell_id="c1", company_id="C1", fiscal_period_id="FY25",
                   metric_id="revenue", result_type=ResultType.COMPARABLE)
    assert c.result_type == ResultType.COMPARABLE
    assert c.execution_status == ExecutionStatus.COMPLETED


def test_partial_matrix_tally():
    cells = [
        CellResult(cell_id="a", company_id="C1", fiscal_period_id="FY25",
                   metric_id="revenue", result_type=ResultType.COMPARABLE,
                   execution_status=ExecutionStatus.COMPLETED),
        CellResult(cell_id="b", company_id="C2", fiscal_period_id="FY25",
                   metric_id="revenue", result_type=ResultType.NOT_EVALUATED,
                   execution_status=ExecutionStatus.NOT_STARTED),
    ]
    m = build_partial_matrix("m1", 2, cells)
    assert m.completed_observation_count == 1
    assert m.completeness_tally == "1/2"
    assert m.execution_status == ExecutionStatus.IN_PROGRESS


def test_partial_matrix_complete():
    cells = [
        CellResult(cell_id="a", company_id="C1", fiscal_period_id="FY25",
                   metric_id="revenue", result_type=ResultType.COMPARABLE,
                   execution_status=ExecutionStatus.COMPLETED),
    ]
    m = build_partial_matrix("m1", 1, cells)
    assert m.execution_status == ExecutionStatus.COMPLETED


def test_cite_originating():
    c = CellResult(cell_id="c1", company_id="C1", fiscal_period_id="FY25",
                   metric_id="revenue", result_type=ResultType.COMPARABLE)
    out = cite_originating(c, "doc-orig", True)
    assert out.originating_document_id == "doc-orig"
    assert out.restated is True


def test_evidence_insufficiency_reason_values():
    assert EvidenceInsufficiencyReason.NO_DOCUMENT.value == "no_document"
    r = EvidenceInsufficiencyReason.CRITIC_VERIFICATION_FAILED
    assert r.value == "critic_verification_failed"


def test_add_cell_increments():
    m = ComparisonMatrix(comparison_id="m", expected_observation_count=2)
    c = CellResult(cell_id="a", company_id="C1", fiscal_period_id="FY25",
                   metric_id="revenue", result_type=ResultType.COMPARABLE,
                   execution_status=ExecutionStatus.COMPLETED)
    m.add_cell(c)
    assert m.completed_observation_count == 1
    assert len(m.cells) == 1
