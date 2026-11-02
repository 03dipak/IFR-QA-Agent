"""Unit tests for LLD-04 tables models."""

from __future__ import annotations

import pytest

from src.tables.models import (
    TableRecord,
    TableCell,
    TableRow,
    TableColumn,
    FootnoteAssertion,
    TableContinuation,
    RowUnitSignal,
    StatementKind,
    ReportingBasis,
    CellKind,
    ScopeType,
)
from src.tables.hard_cases import is_continuation_candidate, validate_no_broadcast, prose_table_split_marker
from src.tables.qa import compute_table_qa, TableQAGroundTruth


def test_table_record_invariants_ok():
    tr = TableRecord(
        table_logical_key="k", table_extraction_id="e", document_id="d",
        extraction_run_id="r", extraction_config_hash="c", freeze_version="v1",
        page_start=1, page_end=2, statement_kind=StatementKind.BALANCE_SHEET,
        reporting_basis=ReportingBasis.CONSOLIDATED,
    )
    assert tr.validate_invariants() == []


def test_table_record_page_span_invalid():
    tr = TableRecord(
        table_logical_key="k", table_extraction_id="e", document_id="d",
        extraction_run_id="r", extraction_config_hash="c", freeze_version="v1",
        page_start=2, page_end=1,
    )
    assert any("page_end" in p for p in tr.validate_invariants())


def test_cell_outside_span():
    cell = TableCell(cell_id="c1", row_id="r1", column_id="col1", raw="100", source_page_index=5)
    row = TableRow(row_id="r1", row_ordinal=1, row_label_raw="Revenue", cells=[cell])
    tr = TableRecord(
        table_logical_key="k", table_extraction_id="e", document_id="d",
        extraction_run_id="r", extraction_config_hash="c", freeze_version="v1",
        page_start=1, page_end=2, rows=[row], columns=[TableColumn(column_id="col1", column_ordinal=1, header_raw="FY25")],
    )
    assert any("outside span" in p for p in tr.validate_invariants())


def test_cell_unknown_column():
    cell = TableCell(cell_id="c1", row_id="r1", column_id="nope", raw="100", source_page_index=1)
    row = TableRow(row_id="r1", row_ordinal=1, row_label_raw="Revenue", cells=[cell])
    tr = TableRecord(
        table_logical_key="k", table_extraction_id="e", document_id="d",
        extraction_run_id="r", extraction_config_hash="c", freeze_version="v1",
        page_start=1, page_end=2, rows=[row], columns=[],
    )
    assert any("unknown column" in p for p in tr.validate_invariants())


def test_make_ids():
    lk = TableRecord.make_logical_key("d", "balance_sheet", "ctx", "hdr")
    ei = TableRecord.make_extraction_id(lk, "cfg", "content")
    assert lk != ei
    assert len(lk) == 64


def test_is_continuation_candidate_confirmed():
    a = TableRecord(table_logical_key="a", table_extraction_id="a1", document_id="d", extraction_run_id="r", extraction_config_hash="c", freeze_version="v1", page_start=1, page_end=1)
    b = TableRecord(table_logical_key="b", table_extraction_id="b1", document_id="d", extraction_run_id="r", extraction_config_hash="c", freeze_version="v1", page_start=2, page_end=2)
    c = is_continuation_candidate(a, b, header_similarity=0.95)
    assert c.relationship_type.value == "confirmed_continuation"


def test_prose_table_split_marker():
    assert prose_table_split_marker("t1") == "[TABLE:t1]"


def test_compute_table_qa():
    truth = [TableQAGroundTruth(table_id="t1", expected_row_count=2, expected_header=["a"], expected_digits=["100"])]
    res = compute_table_qa(truth, {"t1"}, {"t1": 2}, {"t1": True}, {"t1": 0}, {"t1": 1})
    assert res.row_accuracy == 1.0
    assert res.passed is True


def test_footnote_assertion_scope():
    fa = FootnoteAssertion(footnote_id="fn1", text_raw="in lakhs", scope_type=ScopeType.UNRESOLVED)
    assert fa.scope_type == ScopeType.UNRESOLVED


def test_row_unit_signal():
    rus = RowUnitSignal(signal_type="per_share", printed_text="per share", source="row_label")
    assert rus.signal_type == "per_share"
