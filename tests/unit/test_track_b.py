"""Unit tests for the Track B agent layer with the LLM mocked."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from src.agents.critic import Critic
from src.agents.executor import Executor
from src.agents.graph import run_agent
from src.agents.planner import Planner, parse_plan
from src.llmops.tracing import Tracer
from src.tools.retrieve import RetrievedChunk

PLAN_JSON = '{"steps":[{"sub_question":"q1","tool":"retrieve","args":{"query":"q1"}}]}'
CRITIC_PASS = '{"score":5,"issues":[],"verdict":"pass"}'
DOC = "a" * 12


@dataclass
class FakeCompletion:
    text: str
    prompt_tokens: int = 10
    completion_tokens: int = 5
    latency_ms: int = 1


class FakeClient:
    def __init__(self, script: list[str] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[str] = []

    def chat(self, system: str, user: str, max_tokens: int) -> FakeCompletion:
        self.calls.append(system)
        text = self.script.pop(0) if self.script else CRITIC_PASS
        return FakeCompletion(text)


def fake_retriever(**kwargs):
    return [RetrievedChunk(DOC, 3, "Total assets 1,234", 1.0)]


def test_planner_retries_malformed_json_twice_then_fallback():
    client = FakeClient(["not json", "{}", "also bad"])
    plan, completions, retries = Planner(client).plan("question")
    assert len(completions) == 3
    assert retries == 2
    assert [step.tool for step in plan.steps] == ["retrieve"]
    with pytest.raises(ValidationError):
        parse_plan("{}")


def test_executor_tool_exception_becomes_observation():
    def boom(**kwargs):
        raise RuntimeError("downstream failed")

    executor = Executor(tools={"retrieve": boom})
    plan, _, _ = Planner(FakeClient([PLAN_JSON])).plan("question")
    observations = executor.run(plan.steps)
    assert len(observations) == 1
    assert observations[0].error
    assert "RuntimeError" in observations[0].text


def test_executor_step_cap():
    calls: list[str] = []

    def tool(**kwargs):
        calls.append("call")
        return "ok"

    steps = [type("S", (), {"sub_question": f"q{i}", "tool": "retrieve",
                            "args": {"query": f"q{i}"}})() for i in range(15)]
    observations = Executor(tools={"retrieve": tool}, max_steps=3).run(steps)
    assert len(observations) == 3
    assert len(calls) == 3


def test_executor_loop_detection():
    calls: list[str] = []

    def tool(**kwargs):
        calls.append("call")
        return "ok"

    steps = [type("S", (), {"sub_question": f"q{i}", "tool": "retrieve",
                            "args": {"query": "same"}})() for i in range(4)]
    observations = Executor(tools={"retrieve": tool}).run(steps)
    assert len(observations) == 2
    assert observations[-1].error and "loop detected" in observations[-1].text
    assert len(calls) == 1


def test_citation_check_triggers_revise(monkeypatch, tmp_path):
    from src.agents import graph

    monkeypatch.setattr(graph, "resolve_query", lambda q: type(
        "R", (), {"as_dict": lambda self: {}})())
    client = FakeClient([
        PLAN_JSON,
        "Total assets are 1234.",
        CRITIC_PASS,
        "Total assets are 1234 [aaaaaaaaaaaa p.3].",
        CRITIC_PASS,
    ])
    result = run_agent(
        "What were total assets?", client=client,
        executor=Executor(tools={"retrieve": fake_retriever}),
        tracer=Tracer("citation", root=tmp_path / "traces"),
    )
    assert "p.3" in result.answer
    assert result.citations == [f"[{DOC} p.3]"]
    assert any(call.startswith("You are the Reviser") for call in client.calls)


def test_refusal_short_circuits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([])
    result = run_agent("Should I buy this stock?", client=client,
                       tracer=Tracer("refusal", root=tmp_path / "traces"))
    assert result.stopped_reason == "refused"
    assert result.refusal is not None and result.refusal.refused
    assert not client.calls


def test_critic_reuse_validates_schema():
    outcome = Critic(FakeClient(["not json", CRITIC_PASS])).review("q", "draft")
    assert outcome.critique.verdict == "pass"
    assert outcome.parse_retries == 1
