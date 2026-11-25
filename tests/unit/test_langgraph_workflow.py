"""Tests for the LangGraph-backed Track B orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agents import graph
from src.agents.executor import Executor
from src.citations.models import VerificationOutcome
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
    def __init__(self, script: list[str] | None = None, error: str | None = None) -> None:
        self.script = list(script or [])
        self.error = error
        self.calls: list[str] = []

    def chat(self, system: str, user: str, max_tokens: int) -> FakeCompletion:
        self.calls.append(system)
        if self.error:
            raise RuntimeError(self.error)
        return FakeCompletion(self.script.pop(0) if self.script else CRITIC_PASS)


def _stub_resolution(monkeypatch) -> None:
    monkeypatch.setattr(graph, "resolve_query", lambda q: type(
        "R", (), {"as_dict": lambda self: {}})())


def _stub_verification(monkeypatch) -> None:
    monkeypatch.setattr(graph, "direct_verify", lambda *args: VerificationOutcome.VERIFIED)


def _retriever(**_):
    return [RetrievedChunk(DOC, 3, "Total assets 1,234", 1.0)]


def _state(client: FakeClient, tmp_path: Path, *, max_tokens: int = 4000,
           executor: Executor | None = None) -> graph.AgentState:
    return graph.AgentState(
        question="What were total assets?",
        session_id="langgraph-test",
        client=client,
        tracer=Tracer("langgraph-test", root=tmp_path / "traces"),
        executor=executor or Executor(tools={"retrieve": _retriever}),
        max_steps=10,
        max_tokens=max_tokens,
        node_trace=[],
        observations=[],
        steps=[],
        plan_completions=[],
        draft_completions=[],
        revision_completions=[],
        critique_completion=None,
        issues=[],
        remaining_issues=[],
        revision_count=0,
        answer="",
        draft="",
        citations=[],
        stopped_reason="done",
        escalated=False,
    )


def test_compiled_graph_has_required_nodes_and_conditional_edges() -> None:
    compiled = graph.build_agent_graph()
    nodes = set(compiled.get_graph().nodes)
    assert {"intent", "plan", "execute", "synthesize", "critic", "revise", "final"} <= nodes
    edges = compiled.get_graph().edges
    conditional = {(edge.source, edge.target) for edge in edges if edge.conditional}
    assert {
        ("intent", "plan"),
        ("plan", "execute"),
        ("execute", "synthesize"),
        ("critic", "revise"),
        ("critic", "final"),
        ("revise", "critic"),
        ("revise", "final"),
    } <= conditional
    fixed = {(edge.source, edge.target) for edge in edges if not edge.conditional}
    assert ("synthesize", "critic") in fixed
    assert ("final", "__end__") in fixed


def test_langgraph_executes_accepted_path_without_revision(monkeypatch, tmp_path) -> None:
    _stub_resolution(monkeypatch)
    _stub_verification(monkeypatch)
    client = FakeClient([
        PLAN_JSON,
        f"Total assets were 1,234 [{DOC} p.3].",
        CRITIC_PASS,
    ])
    final = graph.build_agent_graph().invoke(_state(client, tmp_path))
    assert final["node_trace"] == ["intent", "plan", "execute", "synthesize", "critic", "final"]
    assert final["stopped_reason"] == "critic_pass"
    assert final["revision_count"] == 0
    assert not any(system.startswith("You are the Reviser") for system in client.calls)
    assert final["citations"] == [f"[{DOC} p.3]"]


def test_langgraph_executes_rejected_path_through_reviser(monkeypatch, tmp_path) -> None:
    _stub_resolution(monkeypatch)
    _stub_verification(monkeypatch)
    client = FakeClient([
        PLAN_JSON,
        "Total assets were 1,234.",
        CRITIC_PASS,
        f"Total assets were 1,234 [{DOC} p.3].",
        CRITIC_PASS,
    ])
    final = graph.build_agent_graph().invoke(_state(client, tmp_path))
    assert final["node_trace"] == [
        "intent", "plan", "execute", "synthesize", "critic", "revise", "final",
    ]
    assert final["revision_count"] == 1
    assert final["stopped_reason"] == "critic_pass"
    assert any(system.startswith("You are the Reviser") for system in client.calls)


def test_conditional_routes_cover_limits_and_termination(monkeypatch) -> None:
    assert graph._route_after_intent({"refusal": object()}) == "end"
    assert graph._route_after_intent({}) == "plan"
    assert graph._route_after_plan({"stopped_reason": "token_budget"}) == "end"
    assert graph._route_after_plan({}) == "execute"
    assert graph._route_after_execute({"stopped_reason": "no_evidence"}) == "end"
    assert graph._route_after_execute({}) == "synthesize"
    assert graph._route_after_critic({"issues": [], "revision_count": 0}) == "final"
    assert graph._route_after_critic({"issues": ["x"], "revision_count": 0}) == "revise"
    assert graph._route_after_critic({"issues": ["x"], "revision_count": 1}) == "final"
    monkeypatch.setattr(graph, "MAX_REVISIONS", 2)
    assert graph._route_after_reviser(
        {"remaining_issues": ["x"], "revision_count": 1},
    ) == "critic"
    monkeypatch.setattr(graph, "MAX_REVISIONS", 1)
    assert graph._route_after_reviser(
        {"remaining_issues": ["x"], "revision_count": 1},
    ) == "final"


def test_reviser_terminates_after_existing_limit(monkeypatch, tmp_path) -> None:
    _stub_resolution(monkeypatch)
    _stub_verification(monkeypatch)
    client = FakeClient([
        PLAN_JSON,
        "Total assets were 1,234.",
        CRITIC_PASS,
        "Total assets were still 1,234.",
    ])
    final = graph.build_agent_graph().invoke(_state(client, tmp_path))
    assert final["node_trace"].count("revise") == 1
    assert final["node_trace"].count("critic") == 1
    assert final["revision_count"] == 1
    assert final["stopped_reason"] == "citation_unresolved"
    assert sum(system.startswith("You are the Reviser") for system in client.calls) == 1


def test_no_evidence_and_token_budget_terminate_early(monkeypatch, tmp_path) -> None:
    _stub_resolution(monkeypatch)
    no_evidence_client = FakeClient([PLAN_JSON])
    no_evidence = graph.build_agent_graph().invoke(_state(
        no_evidence_client, tmp_path,
        executor=Executor(tools={"retrieve": lambda **_: []}),
    ))
    assert no_evidence["node_trace"] == ["intent", "plan", "execute"]
    assert no_evidence["stopped_reason"] == "no_evidence"

    token_client = FakeClient([PLAN_JSON])
    token_budget = graph.build_agent_graph().invoke(_state(
        token_client, tmp_path, max_tokens=15,
    ))
    assert token_budget["node_trace"] == ["intent", "plan"]
    assert token_budget["stopped_reason"] == "token_budget"
    assert token_budget["steps"] == []


def test_run_agent_preserves_public_error_behavior(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_resolution(monkeypatch)
    client = FakeClient(error="llm down")
    result = graph.run_agent(
        "What were total assets?", client=client,
        tracer=Tracer("failure", root=tmp_path / "traces"),
    )
    assert result.stopped_reason == "agent_error"
    assert result.answer == "Agent error: RuntimeError: llm down"
    assert result.tokens == 0
