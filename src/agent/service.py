from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src import request_deadline as budget

from ..llmops.session_log import write_session_log
from ..safety import check_input, mask_value
from .graph import AgentGraph
from .llm import EnvLLM

REQUIRED_ENV = ("LLM_CHAT_URL", "LLM_MODEL", "LLM_API_KEY", "AGENT_RUN_TIMEOUT")
log = logging.getLogger("ifr.service")
_LANGFUSE_WARNINGS: set[str] = set()
_LANGFUSE_CLIENTS: dict[tuple[str, str, str], Langfuse] = {}
_LANGFUSE_FLUSH_TIMEOUT_S = 5.0


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
    trace_id: str = ""


def _trace_id(thread_id: str) -> str:
    return hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:32]


class _LangfuseLLM:
    def __init__(self, delegate: Any, client: Any, model: str) -> None:
        self.delegate = delegate
        self.client = client
        self.model = model

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        if self.client is None:
            return self.delegate.complete(prompt, max_tokens=max_tokens)
        started = time.monotonic()
        with _langfuse_span(self.client.start_as_current_observation(
            name="llm_call",
            as_type="generation",
            model=self.model,
            input={"prompt_chars": len(prompt)},
        )) as span:
            try:
                text = self.delegate.complete(prompt, max_tokens=max_tokens)
            except Exception as exc:
                _update_langfuse_span(
                    span,
                    {
                        "error_type": type(exc).__name__,
                        "latency_s": round(time.monotonic() - started, 3),
                    },
                    "",
                )
                raise
            latency_s = round(time.monotonic() - started, 3)
            estimated_input = max(1, len(prompt) // 4)
            estimated_output = max(1, len(text) // 4)
            _update_langfuse_span(
                span,
                {
                    "latency_s": latency_s,
                    "estimated_tokens": estimated_input + estimated_output,
                    "estimated_input_tokens": estimated_input,
                    "estimated_output_tokens": estimated_output,
                },
                output={"response_chars": len(text)},
                usage_details={
                    "input": estimated_input,
                    "output": estimated_output,
                    "total": estimated_input + estimated_output,
                },
            )
            return text


def _warn_langfuse_once(message: str) -> None:
    if message not in _LANGFUSE_WARNINGS:
        _LANGFUSE_WARNINGS.add(message)
        log.warning(message)


@contextmanager
def _langfuse_span(span_context: Any) -> Any:
    entered = False
    try:
        span = span_context.__enter__()
        entered = True
    except Exception:
        _warn_langfuse_once("Langfuse tracing disabled: span start failed")
        yield None
        return
    try:
        yield span
    except BaseException as exc:
        if entered:
            try:
                span_context.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                _warn_langfuse_once("Langfuse span close failed")
        raise
    else:
        if entered:
            try:
                span_context.__exit__(None, None, None)
            except Exception:
                _warn_langfuse_once("Langfuse span close failed")


def _langfuse_setup(question: str, thread_id: str) -> tuple[Any, Any, Any]:
    public_key = (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    if not public_key or not secret_key:
        _warn_langfuse_once("Langfuse tracing disabled: missing configuration")
        return None, None, nullcontext(None)
    base_url = (os.environ.get("LANGFUSE_BASE_URL") or "").strip()
    cache_key = (public_key, secret_key, base_url)
    try:
        client = _LANGFUSE_CLIENTS.get(cache_key)
        if client is None:
            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url or None,
            )
            _LANGFUSE_CLIENTS[cache_key] = client
        trace_context = {"trace_id": _trace_id(thread_id)}
        handler = CallbackHandler(public_key=public_key, trace_context=trace_context)
        span_context = client.start_as_current_observation(
            name="question",
            as_type="chain",
            trace_context=trace_context,
            input=mask_value(question),
            metadata={"session_id": thread_id, "trace_id": trace_context["trace_id"]},
        )
    except Exception:
        _warn_langfuse_once("Langfuse tracing disabled: setup failed")
        return None, None, nullcontext(None)
    return client, handler, span_context


def _record_langfuse_tool(
    client: Any,
    event: dict[str, Any],
    *,
    latency_s: float,
    token_count: int,
) -> None:
    if client is None or event.get("type") != "tool":
        return
    try:
        input_data = {
            key: event[key]
            for key in ("name", "args_summary", "args")
            if key in event
        }
        with client.start_as_current_observation(
            name=str(event.get("name") or "tool"),
            as_type="tool",
            input=input_data,
            metadata={"latency_s": latency_s, "token_count": token_count},
            usage_details={
                "input": token_count,
                "output": 0,
                "total": token_count,
            },
        ):
            pass
    except Exception:
        _warn_langfuse_once("Langfuse tool span failed")


def _record_langfuse_critic(
    client: Any,
    state: dict[str, Any],
    *,
    latency_s: float,
    token_count: int,
) -> None:
    if client is None:
        return
    critique = state.get("critique") or {}
    feedback = str(critique.get("feedback") or "")
    verdict = str(critique.get("verdict") or "")
    fallback = bool(state.get("fallback_used")) or "fallback" in feedback.lower()
    try:
        with client.start_as_current_observation(
            name="critic",
            as_type="chain",
            input={"citations": mask_value(state.get("citations") or [])},
            output={"verdict": verdict, "fallback": fallback},
            metadata={
                "latency_s": latency_s,
                "token_count": token_count,
                "critic_verdict": verdict,
                "critic_fallback": fallback,
            },
        ):
            pass
    except Exception:
        _warn_langfuse_once("Langfuse critic span failed")


def _update_langfuse_span(
    span: Any,
    metadata: dict[str, Any],
    output: Any,
    usage_details: dict[str, int] | None = None,
) -> None:
    if span is None:
        return
    try:
        if usage_details is None:
            span.update(output=output, metadata=metadata)
        else:
            span.update(output=output, metadata=metadata, usage_details=usage_details)
    except Exception:
        _warn_langfuse_once("Langfuse span update failed")


def _langfuse_metadata(
    thread_id: str,
    stop_reason: str,
    error_type: str,
    latency_s: float,
    token_count: int,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = state or {}
    critique = state.get("critique") or {}
    errors = list(state.get("errors") or [])
    feedback = str(critique.get("feedback") or "")
    metadata = {
        "session_id": thread_id,
        "stop_reason": stop_reason,
        "error_type": error_type,
        "latency_s": latency_s,
        "token_count": token_count,
        "errors": mask_value(errors),
        "critic_verdict": str(critique.get("verdict") or ""),
        "critic_fallback": bool(state.get("fallback_used")) or "fallback" in feedback.lower(),
    }
    return metadata


def _flush_langfuse(client: Any) -> None:
    done = threading.Event()

    def flush() -> None:
        try:
            client.flush()
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=flush, name="langfuse-flush", daemon=True).start()
    if not done.wait(_LANGFUSE_FLUSH_TIMEOUT_S):
        _warn_langfuse_once("Langfuse flush timed out")


def _watch_question_timeout(
    question: str,
    thread_id: str,
    started: float,
    timeout_s: float,
    finished: threading.Event,
    timeout_state: dict[str, bool],
) -> None:
    if finished.wait(timeout_s):
        return
    timeout_state["timed_out"] = True
    _write_log(
        question, "", [], round(time.monotonic() - started, 3), [], thread_id,
        f"TimeoutError: question exceeded {timeout_s:g}s",
        status="timeout",
    )
    timeout_state["logged"] = True


def _formatted_citations(final_answer: dict[str, Any]) -> list[str]:
    pages: dict[str, Any] = {}
    doc_ids: dict[str, str] = {}
    for figure in final_answer.get("figures") or []:
        if not isinstance(figure, dict):
            continue
        cid = str(figure.get("chunk_id") or "")
        if cid:
            if figure.get("page") is not None:
                pages[cid] = figure.get("page")
            if figure.get("document_id"):
                doc_ids[cid] = str(figure.get("document_id"))
    citations: list[str] = []
    seen: set[str] = set()
    for raw in final_answer.get("citations") or []:
        chunk_id = str(raw)
        page = pages.get(chunk_id)
        if page is None:
            m = __import__("re").search(r":p(\d+)$", chunk_id)
            if m:
                page = int(m.group(1))
        doc_id = doc_ids.get(chunk_id) or chunk_id.split(":")[0]
        citation = f"[{doc_id[:12]} p.{page}]" if page is not None else chunk_id
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
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
    status: str = "ok",
) -> None:
    try:
        write_session_log(
            question, answer, citations, latency_s, steps,
            session_id=session_id, error=error,
            stop_reason=stop_reason, error_type=error_type, status=status,
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
        _write_log(question, "", [], 0.0, [], session_id or "default", error, status="error")
        raise MissingEnvironmentError(missing)

    decision = check_input(question)
    safe_question = decision.sanitized
    thread_id = session_id or "default"
    started = time.monotonic()
    timeout_s = float(os.environ["AGENT_RUN_TIMEOUT"])
    if timeout_s <= 0:
        raise ValueError("AGENT_RUN_TIMEOUT must be positive")
    budget.start(timeout_s)
    _write_log(safe_question, "", [], 0.0, [], thread_id, "", status="started")
    if decision.refused:
        latency_s = round(time.monotonic() - started, 3)
        state = {
            "refusal": {
                "refused": True,
                "reason": decision.reason,
                "escalate": decision.escalate,
            },
            "stop_reason": "refused",
            "final_answer": {
                "answer": decision.reason,
                "citations": [],
                "insufficient_evidence": False,
                "figures": [],
            },
        }
        result = QuestionResult(
            state=state,
            answer=decision.reason,
            citations=[],
            raw_citations=[],
            draft_answer=decision.reason,
            session_id=thread_id,
            latency_s=latency_s,
            steps=[],
            events=[],
            token_count=0,
            stop_reason="refused",
            error_type="refused",
            trace_id=_trace_id(thread_id),
        )
        _write_log(
            safe_question, decision.reason, [], latency_s, [], thread_id, "",
            stop_reason="refused", error_type="refused", status="ok",
        )
        return result
    finished = threading.Event()
    timeout_state = {"timed_out": False, "logged": False}
    threading.Thread(
        target=_watch_question_timeout,
        args=(safe_question, thread_id, started, timeout_s, finished, timeout_state),
        name="question-timeout",
        daemon=True,
    ).start()
    state: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    answer = ""
    citations: list[str] = []
    latency_s = 0.0
    stop_reason = ""
    error_type = "none"
    error = ""
    status = "error"
    client: Any = None
    graph: Any = None
    try:
        client, handler, span_context = _langfuse_setup(safe_question, thread_id)
        llm = EnvLLM()
        if client is not None:
            llm = _LangfuseLLM(llm, client, os.environ.get("LLM_MODEL", ""))
        graph = AgentGraph(llm=llm)
        trace_metadata = {
            "langfuse_session_id": thread_id,
            "langfuse_trace_name": "ifr-question",
            "session_id": thread_id,
        }
        with _langfuse_span(span_context) as span:
            try:
                for raw_event in graph.stream(
                    safe_question,
                    thread_id=thread_id,
                    callbacks=[handler] if handler is not None else None,
                    metadata=trace_metadata,
                ):
                    if time.monotonic() - started >= timeout_s or timeout_state["timed_out"]:
                        raise TimeoutError(f"question exceeded {timeout_s:g}s")
                    event = mask_value(dict(raw_event))
                    if event.get("type") == "state":
                        state = dict(event.get("state") or {})
                        continue
                    event["elapsed_s"] = round(time.monotonic() - started, 3)
                    _record_langfuse_tool(
                        client, event,
                        latency_s=event["elapsed_s"],
                        token_count=int(state.get("token_count") or graph.budget.used),
                    )
                    log.info(
                        "stage name=%s kind=%s elapsed_s=%.3f",
                        event.get("node") or event.get("name") or event.get("stage") or "step",
                        event.get("event", ""), event["elapsed_s"],
                    )
                    events.append(event)
                    if on_event is not None:
                        on_event(event)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                status = (
                    "timeout"
                    if isinstance(exc, TimeoutError) or timeout_state["timed_out"]
                    else "error"
                )
                stop_reason = str(state.get("stop_reason") or "tool_error")
                error_type = _error_type(stop_reason, state, error)
                latency_s = round(time.monotonic() - started, 3)
                token_count = int(state.get("token_count") or graph.budget.used)
                _update_langfuse_span(
                    span,
                    _langfuse_metadata(
                        thread_id, stop_reason, error_type, latency_s, token_count, state
                    ),
                    "",
                )
                raise

            final_answer = state.get("final_answer") or {}
            answer = str(final_answer.get("answer") or state.get("draft_answer") or "")
            raw_citations = [str(raw) for raw in final_answer.get("citations") or []]
            citations = _formatted_citations(final_answer)
            latency_s = round(time.monotonic() - started, 3)
            stop_reason = str(state.get("stop_reason") or "")
            token_count = int(state.get("token_count") or graph.budget.used)
            trace_id = _trace_id(thread_id) if client is not None else ""
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
                token_count=token_count,
                stop_reason=stop_reason,
                error_type=_error_type(stop_reason, state, ""),
                trace_id=trace_id,
            )
            _record_langfuse_critic(
                client, state, latency_s=latency_s, token_count=token_count
            )
            _update_langfuse_span(
                span,
                _langfuse_metadata(
                    thread_id, stop_reason, result.error_type, latency_s, token_count, state
                ),
                {"answer": mask_value(answer), "citations": mask_value(citations)},
            )
            status = "timeout" if timeout_state["timed_out"] else "ok"
            return result
    except Exception as exc:
        if not error:
            error = f"{type(exc).__name__}: {exc}"
            status = "timeout" if timeout_state["timed_out"] else "error"
        raise
    finally:
        budget.finish()
        finished.set()
        if not timeout_state["logged"]:
            if not latency_s:
                latency_s = round(time.monotonic() - started, 3)
            _write_log(
                safe_question, mask_value(answer), mask_value(citations), latency_s,
                mask_value(events), thread_id, error,
                stop_reason=stop_reason, error_type=error_type, status=status,
            )
        if client is not None:
            _flush_langfuse(client)
