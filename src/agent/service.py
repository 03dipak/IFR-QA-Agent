from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..llmops.session_log import write_session_log
from .graph import AgentGraph
from .llm import EnvLLM

REQUIRED_ENV = ("LLM_CHAT_URL", "LLM_MODEL", "LLM_API_KEY")
log = logging.getLogger("ifr.service")


class MissingEnvironmentError(RuntimeError):
    def __init__(self, missing: list[str]) -> None:
        self.missing = tuple(missing)
        super().__init__("Missing required environment variables")


@dataclass
class QuestionResult:
    state: dict[str, Any]
    answer: str
    citations: list[str]
    raw_citations: list[str]
    draft_answer: str
    session_id: str
    latency_s: float
    steps: list[dict[str, Any]]
    events: list[dict[str, Any]]
    token_count: int
    stop_reason: str = ""
    error_type: str = "none"


def _formatted_citations(final_answer: dict[str, Any]) -> list[str]:
    pages: dict[str, Any] = {}
    for figure in final_answer.get("figures") or []:
        if isinstance(figure, dict) and figure.get("chunk_id"):
            pages[str(figure["chunk_id"])] = figure.get("page")
    citations: list[str] = []
    for raw in final_answer.get("citations") or []:
        chunk_id = str(raw)
        page = pages.get(chunk_id)
        citations.append(f"[{chunk_id[:12]} p.{page}]" if page is not None else chunk_id)
    return citations


def _error_type(stop_reason: str, state: dict[str, Any], error: str) -> str:
    if error:
        return "tool_error"
    if stop_reason in {"token_budget", "budget"}:
        return "budget"
    if stop_reason in {"reasoning_error"}:
        return "reasoning_error"
    if stop_reason in {"loop_detected", "repeat_query"}:
        return "tool_error"
    final = state.get("final_answer") or {}
    errors = [str(e) for e in state.get("errors") or []]
    lowered_errors = " ".join(errors).lower()
    answer = str(final.get("answer") or "").lower()
    if "company_mismatch" in lowered_errors:
        return "wrong_entity"
    if "basis" in lowered_errors or "period" in lowered_errors:
        return "wrong_period_or_basis"
    if final.get("insufficient_evidence") or any(
        marker in answer
        for marker in ("label not found", "conflict:", "insufficient evidence", "no evidence")
    ):
        return "insufficient_evidence_but_evidence_exists"
    if "unit not" in answer:
        return "unit_mismatch"
    if any(marker in answer for marker in ("reports higher", "difference", "has the larger")):
        return "comparison_arithmetic"
    if any("hallucination" in e.lower() or "ungrounded" in e.lower() for e in errors):
        return "wrong_figure_same_entity"
    if stop_reason in {"step_cap", "revision_cap"}:
        return "budget"
    return "other"


def _write_log(
    question: str,
    answer: str,
    citations: list[str],
    latency_s: float,
    steps: list[dict[str, Any]],
    session_id: str,
    error: str,
    stop_reason: str = "",
    error_type: str = "none",
) -> None:
    try:
        write_session_log(
            question, answer, citations, latency_s, steps,
            session_id=session_id, error=error,
            stop_reason=stop_reason, error_type=error_type,
        )
    except Exception:
        log.exception("session_log_failed session_id=%s", session_id)


def run_question(
    question: str,
    session_id: str | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> QuestionResult:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        error = "Missing required environment variables: " + ", ".join(missing)
        _write_log(question, "", [], 0.0, [], session_id or "default", error)
        raise MissingEnvironmentError(missing)

    thread_id = session_id or "default"
    llm = EnvLLM()
    graph = AgentGraph(llm=llm)
    started = time.monotonic()
    state: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    try:
        for raw_event in graph.stream(question, thread_id=thread_id):
            event = dict(raw_event)
            if event.get("type") == "state":
                state = dict(event.get("state") or {})
                continue
            event["elapsed_s"] = round(time.monotonic() - started, 3)
            events.append(event)
            if on_event is not None:
                on_event(event)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_reason = str(state.get("stop_reason") or "tool_error")
        _write_log(
            question, "", [], time.monotonic() - started, events, thread_id, error,
            stop_reason=stop_reason,
            error_type=_error_type(stop_reason, state, error),
        )
        raise

    final_answer = state.get("final_answer") or {}
    answer = str(final_answer.get("answer") or state.get("draft_answer") or "")
    raw_citations = [str(raw) for raw in final_answer.get("citations") or []]
    citations = _formatted_citations(final_answer)
    latency_s = round(time.monotonic() - started, 3)
    stop_reason = str(state.get("stop_reason") or "")
    result = QuestionResult(
        state=state,
        answer=answer,
        citations=citations,
        raw_citations=raw_citations,
        draft_answer=str(state.get("draft_answer") or ""),
        session_id=thread_id,
        latency_s=latency_s,
        steps=[dict(event) for event in events],
        events=[dict(event) for event in events],
        token_count=int(state.get("token_count") or graph.budget.used),
        stop_reason=stop_reason,
        error_type=_error_type(stop_reason, state, ""),
    )
    _write_log(
        question, answer, citations, latency_s, result.steps, thread_id, "",
        stop_reason=stop_reason, error_type=result.error_type,
    )
    return result
