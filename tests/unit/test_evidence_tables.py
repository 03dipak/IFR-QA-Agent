"""Unit tests for Day 3 evidence-table generation."""

from __future__ import annotations

from src.eval import evidence_tables as et

DOC_A = "a" * 12
DOC_B = "b" * 12
DOC_C = "c" * 12


def _gold(row_id: str = "G-1") -> dict:
    return {
        "row_id": row_id,
        "expected_behaviour": "answer",
        "company_id": "co_a",
        "evidence": [
            {"document_id": DOC_A + "0" * 52, "pdf_page": 10, "evidence_role": "primary"},
            {"document_id": DOC_B + "0" * 52, "pdf_page": 20, "evidence_role": "comparison"},
        ],
    }


def _row(**overrides) -> dict:
    row = {
        "row_id": "G-1", "answer": "Answer 123 [aaaaaaaaaaaa p.10].",
        "answer_correct": False, "stopped_reason": "citation_unresolved",
        "citations": [], "expected_behaviour": "answer",
    }
    row.update(overrides)
    return row


def _call(*chunks, call_index: int = 0) -> dict:
    return {
        "row_id": "G-1", "call_index": call_index, "query": "q",
        "chunks": [
            {"rank": rank, "doc_id": doc, "page": page, "score": 1.0}
            for rank, (doc, page) in enumerate(chunks, start=1)
        ],
    }


def test_first_chunk_parses_trace_detail() -> None:
    detail = "[RetrievedChunk(doc_id='aaaaaaaaaaaa0000', page=387, text='x'), "
    assert et.first_chunk(detail) == (DOC_A, 387)
    assert et.first_chunk("no chunk") is None


def test_diagnostic_refs_handle_multi_document_brackets() -> None:
    answer = (
        "x [aaaaaaaaaaaa p.1, aaaaaaaaaaaa p.2, bbbbbbbbbbbb p.3] "
        "y [doc_id 'cccccccccccc' p.4, p.5]"
    )
    assert et.diagnostic_refs(answer) == [
        (DOC_A, 1), (DOC_A, 2), (DOC_B, 3), (DOC_C, 4), (DOC_C, 5),
    ]


def test_harness_refs_do_not_accept_multi_document_brackets() -> None:
    answer = "[aaaaaaaaaaaa p.1, bbbbbbbbbbbb p.2]"
    assert et.harness_refs(answer) == []
    assert et.diagnostic_refs(answer) == [(DOC_A, 1), (DOC_B, 2)]


def test_tables_a_and_b_use_probe_and_mark_missing_page() -> None:
    gold = {"G-1": _gold()}
    rows = [_row()]
    probes = {"G-1": [_call((DOC_A, 11), (DOC_C, 30))]}
    indexed = {(DOC_A, 10), (DOC_B, 20)}
    a = et.table_a(rows, gold, {}, probes, indexed)
    b = et.table_b(rows, gold, probes, indexed)
    assert a[0]["expected_document_retrieved"] == "yes"
    assert a[0]["retrieval_rank"] == 1
    assert a[1]["expected_document_retrieved"] == "no"
    assert a[1]["missing_document_status"] == "indexed_not_retrieved"
    assert b[0]["correct_page_retrieved"] == "no"
    assert b[0]["page_coverage_status"] == "doc_retrieved_page_missing"
    assert b[1]["page_coverage_status"] == "doc_not_retrieved"


def test_tables_without_probe_are_unknown_not_failures() -> None:
    gold = {"G-1": _gold()}
    rows = [_row()]
    calls = {"G-1": [{"query": "q", "first_chunk": (DOC_A, 10), "error": False}]}
    a = et.table_a(rows, gold, calls, {}, {(DOC_A, 10), (DOC_B, 20)})
    b = et.table_b(rows, gold, {}, {(DOC_A, 10), (DOC_B, 20)})
    assert a[0]["expected_document_retrieved"] == "unknown"
    assert a[0]["missing_document_status"] == "unknown_no_retrieval_evidence"
    assert b[0]["correct_page_retrieved"] == "unknown"


def test_table_c_flags_parsing_failure_and_wrong_page() -> None:
    gold = {"G-1": _gold()}
    answer = "[aaaaaaaaaaaa p.11, bbbbbbbbbbbb p.20]"
    rows = [_row(answer=answer)]
    c = et.table_c(rows, gold, {"G-1": [_call((DOC_A, 11), (DOC_B, 20))]})
    assert c[0]["citation_status"] == "correct_document_wrong_page"
    assert c[0]["failure_category"] == "citation_parsing_failure"
    assert c[1]["citation_status"] == "correct"


def test_table_c_missing_citation_is_not_forced_to_parsing() -> None:
    gold = {"G-1": _gold()}
    c = et.table_c([_row(answer="No citation here.")], gold, {})
    assert c[0]["citation_status"] == "missing_or_unparseable"
    assert c[0]["failure_category"] == "citation_unresolved"


def test_classification_distinguishes_wrong_doc_wrong_page_and_correct_evidence() -> None:
    gold = {"G-1": _gold()}
    indexed = {(DOC_A, 10), (DOC_B, 20)}
    wrong_doc, primary = et.classify_row(_row(), gold["G-1"],
                                         {"G-1": [_call((DOC_C, 1))]}, indexed)
    assert "wrong_document_retrieved" in wrong_doc
    assert primary == "wrong_document_retrieved"

    wrong_page, primary = et.classify_row(_row(), gold["G-1"],
                                          {"G-1": [_call((DOC_A, 11), (DOC_B, 21))]}, indexed)
    assert "correct_document_wrong_page" in wrong_page
    assert "wrong_document_retrieved" not in wrong_page
    assert primary == "correct_document_wrong_page"

    partial, primary = et.classify_row(
        _row(stopped_reason="done"), gold["G-1"],
        {"G-1": [_call((DOC_A, 11))]}, indexed,
    )
    assert "wrong_document_retrieved" in partial
    assert "correct_document_wrong_page" in partial
    assert primary == "wrong_document_retrieved"

    correct, primary = et.classify_row(
        _row(answer_correct=False, stopped_reason="done"), gold["G-1"],
        {"G-1": [_call((DOC_A, 10), (DOC_B, 20))]}, indexed,
    )
    assert "incorrect_answer_despite_correct_evidence" in correct
    assert primary == "incorrect_answer_despite_correct_evidence"


def test_classification_marks_unindexed_gold_page() -> None:
    gold = {"G-1": _gold()}
    found, _ = et.classify_row(_row(), gold["G-1"], {}, {(DOC_A, 10)})
    assert "relevant_evidence_not_retrieved" in found
    assert "other_insufficient_evidence" not in found


def test_classification_does_not_force_when_probe_missing() -> None:
    gold = {"G-1": _gold()}
    found, primary = et.classify_row(_row(answer="", stopped_reason="done"), gold["G-1"], {},
                                     {(DOC_A, 10), (DOC_B, 20)})
    assert primary == "other_insufficient_evidence"
    assert found == ["other_insufficient_evidence"]


def test_summary_denominators_and_null_when_unmeasurable() -> None:
    gold = {"G-1": _gold()}
    rows = [_row()]
    probes = {"G-1": [_call((DOC_A, 11), (DOC_B, 21))]}
    indexed = {(DOC_A, 10), (DOC_B, 20)}
    a = et.table_a(rows, gold, {}, probes, indexed)
    c = et.table_c(rows, gold, probes)
    d = et.table_d(rows, gold, probes, indexed)
    metrics = et.summary(rows, gold, a, c, d, indexed,
                         {"row_ids": ["G-1"], "probes": probes})
    values = {m["metric"]: m for m in metrics["metrics"]}
    assert values["document_retrieval_coverage_rows"]["numerator"] == 1
    assert values["expected_document_retrieval_coverage"]["numerator"] == 2
    assert values["expected_document_retrieval_coverage"]["denominator"] == 2
    assert values["page_retrieval_coverage_rows_any"]["value"] == 0.0
    assert values["expected_page_retrieval_coverage"]["value"] == 0.0
    assert values["gold_index_coverage"]["value"] == 1.0
    assert values["abstention_performance"]["value"] is None


def test_summary_without_probe_reports_null_coverage() -> None:
    gold = {"G-1": _gold()}
    rows = [_row()]
    indexed = {(DOC_A, 10), (DOC_B, 20)}
    a = et.table_a(rows, gold, {}, {}, indexed)
    c = et.table_c(rows, gold, {})
    d = et.table_d(rows, gold, {}, indexed)
    metrics = et.summary(rows, gold, a, c, d, indexed, None)
    values = {m["metric"]: m for m in metrics["metrics"]}
    assert values["document_retrieval_coverage_rows"]["value"] is None
    assert values["page_retrieval_coverage_rows_any"]["value"] is None
