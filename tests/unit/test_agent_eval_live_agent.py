"""Tests for the API-path live agent adapter."""

from src.agent.service import QuestionResult
from src.eval import agent_eval as ae


def test_live_agent_uses_service_path_and_maps_fields(monkeypatch) -> None:
    fake = QuestionResult(
        state={"refusal": {"reason": "out of scope"}},
        answer="final answer",
        citations=["[abcdef123456 p.7]"],
        raw_citations=["[abcdef123456 p.7]"],
        draft_answer="draft",
        session_id="s",
        latency_s=1.234,
        steps=[{"tool": "search_index", "error": False}],
        events=[{"tool": "search_index", "error": False}],
        token_count=7,
        stop_reason="refused",
    )
    monkeypatch.setattr("src.agent.service.run_question", lambda question, session_id=None: fake)

    output = ae.live_agent("question", row={"row_id": "G-001"}, session_id="s")

    assert output.answer == "final answer"
    assert output.citations == ["[abcdef123456 p.7]"]
    assert output.steps == [{"tool": "search_index", "error": False}]
    assert output.tokens == 7
    assert output.latency_ms == 1234
    assert output.stopped_reason == "refused"
    assert output.refusal == "final answer"
