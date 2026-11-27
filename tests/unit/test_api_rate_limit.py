"""Tests for API per-client rate limiting."""

from fastapi.testclient import TestClient

from src.agent.service import QuestionResult
import src.api as api


def test_rate_limit_returns_429(monkeypatch) -> None:
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setattr(api, "warm_index_at_startup", lambda: None)
    monkeypatch.setattr(
        "src.agent.service.run_question",
        lambda question, session_id=None, on_event=None: QuestionResult(
            state={}, answer="ok", citations=[], raw_citations=[], draft_answer="ok",
            session_id=session_id or "s", latency_s=0.1, steps=[], events=[],
            token_count=1, stop_reason="accepted", error_type="none",
        ),
    )
    client = TestClient(api.app)
    first = client.post("/ask", json={"question": "What is total assets?"})
    second = client.post("/ask", json={"question": "What is total assets?"})
    assert first.status_code == 200
    assert second.status_code == 429
