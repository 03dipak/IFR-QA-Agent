"""Unit tests for the reflection loop using a fake LLM client (no network)."""

from __future__ import annotations

import json

from src.agents.llm_client import Completion
from src.agents.reflection_loop import run_reflection

GEN_SYSTEM_PREFIX = "You are the Generator"
CRITIC_SYSTEM_PREFIX = "You are the Critic"
REVISER_SYSTEM_PREFIX = "You are the Reviser"


class FakeClient:
    """Scripted fake: dispatches on the system prompt role marker."""

    def __init__(
        self,
        generator_texts: list[str],
        critic_texts: list[str],
        reviser_texts: list[str] | None = None,
        tokens: tuple[int, int] = (10, 20),
    ) -> None:
        self._gen = iter(generator_texts)
        self._critic = iter(critic_texts)
        self._rev = iter(reviser_texts or [])
        self._tokens = tokens
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, max_tokens: int) -> Completion:
        self.calls.append((system, user))
        if system.startswith(CRITIC_SYSTEM_PREFIX):
            text = next(self._critic)
        elif system.startswith(REVISER_SYSTEM_PREFIX):
            text = next(self._rev)
        elif system.startswith(GEN_SYSTEM_PREFIX):
            text = next(self._gen)
        else:
            raise AssertionError(f"unknown system prompt: {system[:40]}")
        return Completion(text, self._tokens[0], self._tokens[1], latency_ms=5)


def _critic_json(score: int, verdict: str, issues: list[str] | None = None) -> str:
    return json.dumps({"score": score, "issues": issues or [], "verdict": verdict})


def test_pass_on_first_try() -> None:
    client = FakeClient(["draft one"], [_critic_json(5, "pass")])
    result = run_reflection(client, "What is revenue?", max_iterations=2)
    assert result.passed is True
    assert result.stopped_reason == "critic_pass"
    assert result.iterations == 1
    assert result.final_answer == "draft one"
    assert [step.stage for step in result.trace] == ["generate", "critique"]
    assert result.total_tokens == 60


def test_revise_then_pass() -> None:
    client = FakeClient(
        ["draft one"],
        [_critic_json(3, "revise", ["missing year"]), _critic_json(4, "pass")],
        ["revised draft"],
    )
    result = run_reflection(client, "What is revenue?", max_iterations=2)
    assert result.passed is True
    assert result.iterations == 2
    assert result.final_answer == "revised draft"
    assert [step.stage for step in result.trace] == ["generate", "critique", "revise", "critique"]
    assert result.trace[2].stage == "revise"


def test_max_iterations_stop() -> None:
    client = FakeClient(
        ["draft one"],
        [_critic_json(2, "revise"), _critic_json(2, "revise")],
        ["revised draft"],
    )
    result = run_reflection(client, "What is revenue?", max_iterations=2)
    assert result.passed is False
    assert result.stopped_reason == "max_iterations"
    assert result.iterations == 2
    assert [step.stage for step in result.trace] == ["generate", "critique", "revise", "critique"]


def test_malformed_critic_json_fallback() -> None:
    client = FakeClient(["draft one"], ["not json at all", "still not json"])
    result = run_reflection(client, "What is revenue?", max_iterations=2)
    assert result.passed is True
    assert result.stopped_reason == "critic_pass"
    assert result.iterations == 1
    critique = result.trace[1].critique
    assert critique is not None and critique["verdict"] == "pass"


def test_malformed_then_valid_revise() -> None:
    client = FakeClient(
        ["draft one"],
        ["oops", _critic_json(3, "revise", ["bad number"]), _critic_json(5, "pass")],
        ["fixed draft"],
    )
    result = run_reflection(client, "What is revenue?", max_iterations=2)
    assert result.passed is True
    assert result.iterations == 2
    assert result.trace[1].critique is not None
    assert result.trace[1].critique["score"] == 3


def test_token_budget_cap() -> None:
    client = FakeClient(
        ["draft one"],
        [_critic_json(2, "revise"), _critic_json(2, "revise")],
        ["revised draft"],
    )
    result = run_reflection(client, "q", max_iterations=3, max_total_tokens=90)
    assert result.passed is False
    assert result.stopped_reason == "token_budget"
    assert result.total_tokens <= 90 + 30
