"""Track B agent graph: planner -> executor -> generator -> critic/reviser with citations."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.critic import Critic, Critique
from src.agents.executor import Executor, Observation
from src.agents.generator import Generator
from src.agents.llm_client import Completion, EndpointLLMClient, LLMClient
from src.agents.planner import Plan, Planner
from src.agents.reviser import Reviser
from src.citations.models import EvidenceObject, EvidenceType, PageRef, VerificationOutcome
from src.citations.verification import direct_verify
from src.llmops.tracing import Tracer
from src.resolver.query_resolver import resolve_query
from src.safety.refusal import Refusal, check_evidence, check_intent

_CITATION_RE = re.compile(
    r"\[(?:doc_id\s*:?\s*)?['\"]?([0-9a-f]{8,64})['\"]?\s+((?:p\.\s*\d+\s*,?\s*)+)\]",
    re.I,
)
_CITATION_PAGE_RE = re.compile(r"p\.\s*(\d+)", re.I)
MAX_STEPS = 10
MAX_TOKENS = 4000
MAX_EVIDENCE_CHARS = 3600
MAX_CHUNK_CHARS = 600
MAX_REVISIONS = 1


@dataclass
class AgentResult:
    answer: str
    citations: list[str]
    steps: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    latency_ms: int = 0
    session_id: str = ""
    stopped_reason: str = "done"
    refusal: Refusal | None = None
    escalated: bool = False


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    client: LLMClient
    tracer: Tracer
    executor: Executor | None
    max_steps: int
    max_tokens: int
    node_trace: list[str]
    resolution: Any
    plan: Plan | None
    observations: list[Observation]
    steps: list[dict[str, Any]]
    plan_completions: list[Completion]
    draft_completions: list[Completion]
    critique_completion: Completion | None
    revision_completions: list[Completion]
    critique: Critique | None
    draft: str
    answer: str
    citations: list[str]
    issues: list[str]
    remaining_issues: list[str]
    revision_count: int
    stopped_reason: str
    refusal: Refusal | None
    escalated: bool


def _usage(completions: list[Completion]) -> int:
    return sum(c.prompt_tokens + c.completion_tokens for c in completions)


def _citation_refs(answer: str) -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    for match in _CITATION_RE.finditer(answer or ""):
        doc_id = match.group(1)[:12]
        for page in _CITATION_PAGE_RE.findall(match.group(2)):
            refs.append((doc_id, int(page)))
    return list(dict.fromkeys(refs))


def _build_evidence_text(observations: list[Observation]) -> str:
    parts: list[str] = []
    for observation in observations:
        if observation.chunks:
            for chunk in observation.chunks:
                parts.append(f"[{chunk.doc_id[:12]} p.{chunk.page}] {chunk.text[:MAX_CHUNK_CHARS]}")
        elif observation.text:
            parts.append(observation.text[:MAX_CHUNK_CHARS])
    return "\n".join(parts)[:MAX_EVIDENCE_CHARS]


def _citation_check(answer: str, observations: list[Observation]) -> tuple[list[str], list[str]]:
    available = {
        (chunk.doc_id[:12], chunk.page): chunk.text
        for observation in observations for chunk in observation.chunks
    }
    citations = [f"[{doc_id} p.{page}]" for doc_id, page in _citation_refs(answer)]
    issues = []
    for key in _citation_refs(answer):
        page_text = available.get(key)
        if not page_text:
            issues.append(f"citation {key[0]} p.{key[1]} is not in retrieved chunks")
            continue
        evidence = EvidenceObject(
            evidence_id=EvidenceObject.make_evidence_id(key[0], key[1], ""),
            evidence_type=EvidenceType.PROSE, corpus_freeze_id="freeze_v1",
            document_id=key[0], page=PageRef(pdf_page_index=key[1]),
        )
        outcome = direct_verify(evidence, page_text, "", "", "", "")
        if outcome is not VerificationOutcome.VERIFIED:
            issues.append(f"citation {key[0]} p.{key[1]} failed verification: {outcome.value}")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", answer) if s.strip()]
    factual = [s for s in sentences if not s.startswith("[") and len(s.split()) > 3]
    if factual and not citations:
        issues.append("answer has factual statements with no [doc_id p.N] citation")
    return citations, issues


def _node_update(state: AgentState, node: str, **updates: Any) -> dict[str, Any]:
    updates["node_trace"] = [*state.get("node_trace", []), node]
    return updates


def _steps_from_observations(observations: list[Observation]) -> list[dict[str, Any]]:
    return [
        {"step": o.step, "tool": o.tool, "args": o.args, "latency_ms": o.latency_ms,
         "error": o.error, "observation": o.text[:300]}
        for o in observations
    ]


def _intent_node(state: AgentState) -> dict[str, Any]:
    refusal = check_intent(state["question"])
    state["tracer"].record("intent", detail="intent_checked", refused=refusal.refused)
    if refusal.refused:
        return _node_update(
            state, "intent", refusal=refusal, escalated=refusal.escalate,
            answer=f"I cannot answer: {refusal.reason}", stopped_reason="refused",
        )
    return _node_update(state, "intent")


def _plan_node(state: AgentState) -> dict[str, Any]:
    resolution = resolve_query(state["question"])
    plan, completions, retries = Planner(state["client"]).plan(state["question"])
    state["tracer"].record("plan", tool="planner", detail="plan_completed",
                           step_count=len(plan.steps),
                           prompt_tokens=_usage(completions), malformed_retries=retries)
    plan_completions = [*state.get("plan_completions", []), *completions]
    stopped_reason = "token_budget" if _usage(plan_completions) >= state["max_tokens"] else ""
    return _node_update(
        state, "plan", resolution=resolution, plan=plan, plan_completions=plan_completions,
        stopped_reason=stopped_reason,
    )


def _execute_node(state: AgentState) -> dict[str, Any]:
    plan = state["plan"]
    assert plan is not None
    runner = state.get("executor") or Executor(
        max_steps=state["max_steps"], max_tokens=state["max_tokens"], tracer=state["tracer"],
    )
    observations = runner.run(plan.steps, tokens_used=_usage(state.get("plan_completions", [])))
    steps = _steps_from_observations(observations)
    if not any(observation.chunks for observation in observations):
        refusal = check_evidence(False)
        if refusal.refused:
            return _node_update(
                state, "execute", observations=observations, steps=steps, refusal=refusal,
                escalated=refusal.escalate, answer=f"I cannot answer: {refusal.reason}",
                stopped_reason="no_evidence",
            )
    return _node_update(state, "execute", observations=observations, steps=steps)


def _synthesis_node(state: AgentState) -> dict[str, Any]:
    observations = state.get("observations", [])
    evidence_text = _build_evidence_text(observations)
    draft, completion = Generator(state["client"]).generate(
        f"Question: {state['question']}\n\nTool observations:\n{evidence_text}\n"
        "Answer concisely; cite every claim as [doc_id p.N].",
    )
    state["tracer"].record("generate", tool="generator", detail="draft_generated",
                           prompt_tokens=completion.prompt_tokens,
                           completion_tokens=completion.completion_tokens)
    citations, issues = _citation_check(draft, observations)
    return _node_update(
        state, "synthesize", draft=draft, answer=draft, citations=citations, issues=issues,
        draft_completions=[*state.get("draft_completions", []), completion],
    )


def _critic_node(state: AgentState) -> dict[str, Any]:
    outcome = Critic(state["client"]).review(state["question"], state.get("draft", ""))
    state["tracer"].record("critique", tool="critic", detail="critique_completed",
                           prompt_tokens=outcome.completion.prompt_tokens,
                           completion_tokens=outcome.completion.completion_tokens)
    issues = list(state.get("issues", []))
    if outcome.critique.verdict == "revise":
        issues.extend(outcome.critique.issues)
    completions = [
        *state.get("plan_completions", []), *state.get("draft_completions", []),
        outcome.completion,
    ]
    stopped_reason = state.get("stopped_reason", "")
    if _usage(completions) >= state["max_tokens"]:
        stopped_reason = "token_budget"
    return _node_update(
        state, "critic", critique=outcome.critique, critique_completion=outcome.completion,
        issues=issues, stopped_reason=stopped_reason,
    )


def _revise_node(state: AgentState) -> dict[str, Any]:
    revised, revision = Reviser(state["client"]).revise(
        state["question"], state.get("draft", ""), state.get("issues", []),
    )
    observations = state.get("observations", [])
    citations, remaining = _citation_check(revised, observations)
    state["tracer"].record("revise", tool="reviser", detail="revision_completed",
                           prompt_tokens=revision.prompt_tokens,
                           completion_tokens=revision.completion_tokens)
    stopped_reason = state.get("stopped_reason", "")
    if remaining:
        stopped_reason = "citation_unresolved"
    return _node_update(
        state, "revise", answer=revised, citations=citations, remaining_issues=remaining,
        revision_count=state.get("revision_count", 0) + 1,
        revision_completions=[*state.get("revision_completions", []), revision],
        stopped_reason=stopped_reason,
    )


def _final_synthesis_node(state: AgentState) -> dict[str, Any]:
    return _node_update(
        state, "final", answer=state.get("answer", state.get("draft", "")),
        citations=state.get("citations", []),
        stopped_reason=state.get("stopped_reason") or "critic_pass",
    )


def _route_after_intent(state: AgentState) -> str:
    return "end" if state.get("refusal") is not None else "plan"


def _route_after_plan(state: AgentState) -> str:
    return "end" if state.get("stopped_reason") == "token_budget" else "execute"


def _route_after_execute(state: AgentState) -> str:
    return "end" if state.get("stopped_reason") == "no_evidence" else "synthesize"


def _route_after_critic(state: AgentState) -> str:
    if state.get("issues") and state.get("revision_count", 0) < MAX_REVISIONS:
        return "revise"
    return "final"


def _route_after_reviser(state: AgentState) -> str:
    if state.get("remaining_issues") and state.get("revision_count", 0) < MAX_REVISIONS:
        return "critic"
    return "final"


def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("intent", _intent_node)
    builder.add_node("plan", _plan_node)
    builder.add_node("execute", _execute_node)
    builder.add_node("synthesize", _synthesis_node)
    builder.add_node("critic", _critic_node)
    builder.add_node("revise", _revise_node)
    builder.add_node("final", _final_synthesis_node)
    builder.set_entry_point("intent")
    builder.add_conditional_edges(
        "intent", _route_after_intent, {"plan": "plan", "end": END},
    )
    builder.add_conditional_edges(
        "plan", _route_after_plan, {"execute": "execute", "end": END},
    )
    builder.add_conditional_edges(
        "execute", _route_after_execute, {"synthesize": "synthesize", "end": END},
    )
    builder.add_edge("synthesize", "critic")
    builder.add_conditional_edges(
        "critic", _route_after_critic, {"revise": "revise", "final": "final"},
    )
    builder.add_conditional_edges(
        "revise", _route_after_reviser, {"critic": "critic", "final": "final"},
    )
    builder.add_edge("final", END)
    return builder.compile()


def run_agent(
    question: str,
    *,
    client: LLMClient | None = None,
    session_id: str | None = None,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
    executor: Executor | None = None,
    tracer: Tracer | None = None,
) -> AgentResult:
    start = time.monotonic()
    session_id = session_id or uuid.uuid4().hex[:12]
    tracer = tracer or Tracer(session_id)
    client = client or EndpointLLMClient()
    result = AgentResult(answer="", citations=[], session_id=session_id)
    state: AgentState = {
        "question": question,
        "session_id": session_id,
        "client": client,
        "tracer": tracer,
        "executor": executor,
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "node_trace": [],
        "observations": [],
        "steps": [],
        "plan_completions": [],
        "draft_completions": [],
        "revision_completions": [],
        "critique_completion": None,
        "issues": [],
        "remaining_issues": [],
        "revision_count": 0,
        "answer": "",
        "draft": "",
        "citations": [],
        "stopped_reason": "done",
        "escalated": False,
    }
    final_state = state
    try:
        final_state = build_agent_graph().invoke(state)
        result.answer = final_state.get("answer", "")
        result.citations = final_state.get("citations", [])
        result.refusal = final_state.get("refusal")
        result.escalated = final_state.get("escalated", False)
        result.stopped_reason = final_state.get("stopped_reason", "done")
    except Exception as exc:
        result.answer = f"Agent error: {type(exc).__name__}: {exc}"
        result.stopped_reason = "agent_error"
        tracer.record("error", detail=type(exc).__name__)

    result.steps = final_state.get("steps", [])
    tracked = [
        *final_state.get("plan_completions", []),
        *final_state.get("draft_completions", []),
        *final_state.get("revision_completions", []),
    ]
    critique_completion = final_state.get("critique_completion")
    if critique_completion is not None:
        tracked.append(critique_completion)
    result.tokens = _usage(tracked)
    result.latency_ms = int((time.monotonic() - start) * 1000)
    if result.tokens >= max_tokens:
        result.stopped_reason = "token_budget"
    tracer.record("result", detail=result.stopped_reason, tokens=result.tokens,
                  latency_ms=result.latency_ms)
    return result
