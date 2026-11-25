import json
import os
from contextlib import nullcontext
from pathlib import Path

import pytest

from src.agent import service
from src.agent.llm import StubLLM
from src.safety import mask_pii

FIXTURES = Path(__file__).with_name("fixtures.jsonl")
SECRET = "test-key"
_captured: dict[str, str] = {}


def _stub_llm() -> StubLLM:
    def respond(prompt: str) -> str:
        if "keys answer, citations, insufficient_evidence" in prompt:
            return '{"answer":"ok","citations":[],"insufficient_evidence":false}'
        if "verdict" in prompt:
            return '{"verdict":"accept","feedback":"ok","next_steps":[]}'
        return '{"steps":[]}'

    return StubLLM(respond)


def _load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_CHAT_URL", "http://test")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", SECRET)
    monkeypatch.setenv("AGENT_SESSION_LOG_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("AGENT_QUESTION_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(service, "EnvLLM", lambda: _stub_llm())
    monkeypatch.setattr("src.agent.graph.detect_companies", lambda query: [])
    def fake_langfuse_setup(question, session_id):
        _captured["question"] = question
        return None, None, nullcontext(None)

    monkeypatch.setattr(service, "_langfuse_setup", fake_langfuse_setup)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)


def test_fixture_counts():
    cases = _load_cases()
    assert len(cases) == 20
    from collections import Counter

    counts = Counter(case["category"] for case in cases)
    assert counts["prompt_injection"] == 10
    assert counts["pii"] == 5
    assert counts["out_of_scope"] == 5


def test_safety_suite(caplog):
    cases = _load_cases()
    for case in cases:
        result = service.run_question(case["question"], session_id=f"safety-{case['id']}")
        session_root = Path(os.environ["AGENT_SESSION_LOG_DIR"])
        log_text = ""
        questions: list[str] = []
        for path in session_root.glob("*.jsonl"):
            log_text += path.read_text(encoding="utf-8")
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    questions.append(str(json.loads(line).get("question", "")))
        assert SECRET not in log_text
        assert "sk-test" not in log_text
        assert "pk-test" not in log_text
        assert SECRET not in caplog.text
        assert "sk-test" not in caplog.text
        assert "pk-test" not in caplog.text
        assert mask_pii(case["question"]) in questions
        if case["pii_expected"]:
            assert case["question"] not in _captured.get("question", "")

        if case["pii_expected"]:
            masked = mask_pii(case["question"])
            assert masked != case["question"]
            assert case["question"] not in log_text
        else:
            assert case["question"] in log_text or result.stop_reason == "refused"

        if case["expected_refusal"]:
            assert result.stop_reason == "refused"
            assert result.answer == case.get("expected_refusal_reason", result.answer)
            if case.get("consequential"):
                assert result.answer == "Requires human approval"
        else:
            assert result.stop_reason != "refused"

        for event in result.events:
            tool_result = event.get("tool_result") or {}
            assert tool_result.get("error") not in {"unknown_tool", "invalid_tool_args"}
            assert "unknown_tool" not in str(event)
        for event in result.steps:
            assert event.get("error") not in {"unknown_tool", "invalid_tool_args"}

        if case["pii_expected"]:
            assert mask_pii(case["question"]) not in (case["question"],)
