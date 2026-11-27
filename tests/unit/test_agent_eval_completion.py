"""Tests for strict completion scoring."""

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


def test_completed_requires_clean_accepted_critic() -> None:
    row = _row()
    clean = ae.AgentOutput(
        answer="answer", stopped_reason="accepted", errors=[],
        critic_passed=True, critic_fallback=False,
    )
    assert ae.score_row(row, clean)["task_completed"] is True

    assert not ae.score_row(row, ae.AgentOutput(
        answer="answer", stopped_reason="accepted", errors=["boom"],
        critic_passed=True, critic_fallback=False,
    ))["task_completed"]
    assert not ae.score_row(row, ae.AgentOutput(
        answer="answer", stopped_reason="accepted", errors=[],
        critic_passed=True, critic_fallback=True,
    ))["task_completed"]
    assert not ae.score_row(row, ae.AgentOutput(
        answer="answer", stopped_reason="accepted", errors=[],
        critic_passed=False, critic_fallback=False,
    ))["task_completed"]


def test_critic_fallback_uses_state_or_explicit_error_only() -> None:
    assert not ae._critic_fallback_from_state(
        {"fallback_used": False, "critique": {"feedback": "safe fallback"}}, []
    )
    assert ae._critic_fallback_from_state({"fallback_used": True}, [])
    assert ae._critic_fallback_from_state({}, ["critic: fallback used"])


def test_completion_split_distinguishes_llm_and_deterministic_accept() -> None:
    row = _row()
    llm = ae.score_row(row, ae.AgentOutput(
        answer="answer", stopped_reason="accepted", errors=[],
        critic_passed=True, critic_fallback=False,
        completion_kind="llm_critic_pass",
    ))
    deterministic = ae.score_row(row, ae.AgentOutput(
        answer="answer", stopped_reason="accepted", errors=[],
        critic_passed=True, critic_fallback=False,
        completion_kind="deterministic_accept",
    ))
    incomplete = ae.score_row(row, ae.AgentOutput(answer="", stopped_reason="accepted"))
    metrics = ae.aggregate([llm, deterministic, incomplete])
    assert metrics["task_completion_rate"] == 0.6667
    assert metrics["task_completion_split"] == {
        "llm_critic_pass": 1, "deterministic_accept": 1, "legacy": 0, "none": 1,
    }
