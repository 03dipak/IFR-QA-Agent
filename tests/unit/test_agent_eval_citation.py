"""Tests for API citation mapping in the eval harness."""

from src.eval import agent_eval as ae


def _row(**overrides):
    row = {
        "row_id": "G-001", "question_id": "Q-001", "split": "dev",
        "expected_behaviour": "answer", "expected_answer": "1,234.56 crore",
        "document_id": "a" * 12,
        "evidence": [{"document_id": "a" * 12, "pdf_page": 7}],
    }
    row.update(overrides)
    return row


def test_citation_correct_uses_returned_citation() -> None:
    row = _row()
    assert ae.citation_correct(row, "answer without citation", ["[aaaaaaaaaaaa p.7]"])


def test_citation_correct_rejects_wrong_page() -> None:
    row = _row()
    assert not ae.citation_correct(row, "answer without citation", ["[aaaaaaaaaaaa p.8]"])


def test_citation_correct_rejects_no_citation() -> None:
    row = _row()
    assert not ae.citation_correct(row, "answer without citation", [])


def test_score_row_marks_failed_run() -> None:
    row = _row()
    output = ae.AgentOutput(
        answer="", citations=[], prompt_tokens=1, completion_tokens=0, tokens=1,
        latency_ms=1, steps=[{"tool": "retrieve", "error": "boom"}],
        stopped_reason="agent_error",
    )
    scored = ae.score_row(row, output)
    assert "tool_failure" in scored["failures"]
    assert "incomplete_task" in scored["failures"]


def test_recall_at_k_uses_saved_retrieved_pages() -> None:
    row = _row()
    output = ae.AgentOutput(
        answer="answer", citations=["[aaaaaaaaaaaa p.7]"],
        retrieved_doc_pages=["[aaaaaaaaaaaa p.7]"], recall_k=5,
    )
    scored = ae.score_row(row, output)
    assert scored["recall_at_k"] is True
    assert scored["retrieved_doc_pages"] == ["[aaaaaaaaaaaa p.7]"]
    assert scored["trace_id"] == ""
