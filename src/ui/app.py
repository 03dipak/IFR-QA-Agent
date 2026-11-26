"""Streamlit chat UI for the IFR-QA Agent (lld_api_ui.txt section 3)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_BACKEND_URL = os.environ.get(
    "BACKEND_URL", os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")
)
REQUEST_TIMEOUT = float(os.environ.get("BACKEND_TIMEOUT_SECONDS", "300"))
SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "docs" / "DEMO_SCENARIOS.md"
STAGE_LABELS = {
    "request": "Request received",
    "intent": "Request received",
    "plan": "Agent planning",
    "retrieval": "Document/page retrieval",
    "calculation": "Calculator execution",
    "tool": "Tool execution",
    "generate": "Answer generation",
    "critique": "Answer review",
    "revise": "Answer revision",
    "result": "Agent execution completed",
    "completed": "Completed",
    "error": "Request failed",
}
ICONS = {"running": "●", "completed": "✓", "failed": "✗"}
STEP_LABELS = {
    "planner": "Planning...",
    "executor": "Searching...",
    "critic": "Checking evidence...",
    "answer": "Writing answer...",
}


def step_status_label(event: dict[str, Any]) -> str:
    node = str(event.get("node", ""))
    return STEP_LABELS.get(node, str(event.get("detail") or "Working..."))


def tool_status_label(event: dict[str, Any]) -> str:
    name = str(event.get("name", ""))
    summary = str(event.get("args_summary", "")).strip()
    if name in {"search_index", "get_chunk"}:
        return f"Searching: {summary}" if summary else "Searching..."
    if name == "calculator":
        return f"Calculating: {summary}" if summary else "Calculating..."
    return f"Using {name}: {summary}" if summary else f"Using {name}..."


def load_examples() -> list[tuple[str, str]]:
    if not SCENARIOS_PATH.exists():
        return []
    examples: list[tuple[str, str]] = []
    for line in SCENARIOS_PATH.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0].isdigit():
            examples.append((f"{cells[1]} ({cells[2]})", cells[3]))
    return examples


def initial_progress() -> dict[str, dict[str, str]]:
    return {}


def add_user_message(messages: list[dict[str, Any]], question: str) -> None:
    messages.append({"role": "user", "content": question})


def add_assistant_message(
    messages: list[dict[str, Any]], *, data: dict[str, Any] | None = None, error: str = ""
) -> None:
    if error:
        messages.append({"role": "assistant", "error": error})
    else:
        messages.append({"role": "assistant", "data": data or {}})


def _upsert_stage(progress: dict[str, dict[str, str]], key: str, label: str, status: str) -> None:
    if key in progress:
        progress[key]["status"] = status
        return
    progress[key] = {"label": label, "status": status}


def apply_progress_event(progress: dict[str, dict[str, str]], event: dict[str, Any]) -> None:
    kind = event.get("event")
    if kind == "progress":
        stage = str(event.get("stage", ""))
        label = STAGE_LABELS.get(stage)
        status = str(event.get("status", ""))
        if not label or status not in ICONS:
            return
        key = "request" if stage == "intent" else stage
        _upsert_stage(progress, key, label, status)
    elif kind == "error":
        if "request" in progress:
            progress["request"]["status"] = "failed"
        _upsert_stage(progress, "error", STAGE_LABELS["error"], "failed")
    elif kind == "final":
        if "request" in progress:
            progress["request"]["status"] = "completed"
        _upsert_stage(progress, "completed", STAGE_LABELS["completed"], "completed")


def progress_markdown(progress: dict[str, dict[str, str]]) -> str:
    return "\n".join(
        f"{ICONS[item['status']]} **{item['label']}**" for item in progress.values()
    )


def parse_sse(response: httpx.Response) -> Iterator[dict[str, Any]]:
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            yield {"event": "error", "message": "Invalid JSON from API"}
            return
        if isinstance(event, dict):
            yield event


def stream_answer(backend_url: str, question: str, timeout: float) -> Iterator[dict[str, Any]]:
    with httpx.stream(
        "POST",
        f"{backend_url.rstrip('/')}/ask/stream",
        json={"question": question},
        timeout=timeout,
    ) as response:
        if response.status_code >= 400:
            response.read()
        response.raise_for_status()
        yield from parse_sse(response)


@st.cache_data(ttl=1)
def backend_ready(backend_url: str) -> tuple[bool, str]:
    try:
        response = httpx.get(f"{backend_url.rstrip('/')}/ready", timeout=3)
        response.raise_for_status()
        data = response.json()
        return bool(data.get("ready")), str(data.get("error") or "")
    except Exception as exc:
        return False, f"cannot reach backend: {type(exc).__name__}: {exc}"


def is_conversational(data: dict[str, Any]) -> bool:
    return str(data.get("response_type", "financial")) == "conversational"


def apply_final_event(progress: dict[str, dict[str, str]], event: dict[str, Any]) -> None:
    apply_progress_event(progress, event)
    if is_conversational(event.get("data") or {}):
        progress.clear()


def render_status(data: dict[str, Any]) -> None:
    refusal = data.get("refusal") or {}
    stopped = str(data.get("stopped_reason", ""))
    if refusal.get("refused") or stopped == "refused":
        st.warning(f"Refusal: {refusal.get('reason', 'out of scope')}")
    elif stopped == "no_evidence":
        st.warning("Insufficient in-corpus evidence; no financial figure was returned.")
    citation_status = str(data.get("citation_status", ""))
    if citation_status in {"unresolved", "partially_unresolved"}:
        st.warning(f"Unresolved citations: status={citation_status}; verify evidence below.")
    elif citation_status == "no_citations" and stopped not in {"no_evidence", "refused"}:
        st.info("No page citations were returned with this answer.")


def render_citations(citations: list[str]) -> None:
    with st.expander("Citations (chunk id + page)"):
        for citation in citations:
            text = str(citation).strip().strip("[]")
            chunk_id, _, page = text.partition(" p.")
            suffix = f" — page {page}" if page else ""
            st.markdown(f"- `{chunk_id}`{suffix}")


def render_evidence(evidence: list[dict[str, Any]]) -> None:
    for index, item in enumerate(evidence, start=1):
        st.markdown(
            f"**{index}. {item.get('citation', '')}** "
            f"`{item.get('document_id', '')}` p.{item.get('pdf_page', '')}"
        )
        if item.get("text"):
            st.caption(item["text"])


def render_calculations(calculations: list[dict[str, Any]]) -> None:
    for index, item in enumerate(calculations, start=1):
        st.markdown(f"**{index}.** `{item.get('expression', '')}` = `{item.get('value', '')}`")
        if item.get("status"):
            st.caption(f"status: {item['status']}")


def safe_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "tool": step.get("tool", ""),
            "status": "failed" if step.get("error") else "ok",
            "latency_ms": step.get("latency_ms", 0),
        }
        for step in trace
    ]


def render_answer(data: dict[str, Any]) -> None:
    st.markdown(data.get("answer", ""))
    if is_conversational(data):
        return
    render_status(data)
    citations = list(data.get("citations") or [])
    if citations:
        render_citations(citations)
    evidence = list(data.get("evidence") or [])
    if evidence:
        with st.expander("Supporting evidence (document IDs + PDF pages)"):
            render_evidence(evidence)
    calculations = list(data.get("calculations") or [])
    if calculations:
        with st.expander("Calculation results"):
            render_calculations(calculations)
    trace = list(data.get("trace") or [])
    if trace:
        with st.expander("Execution trace"):
            st.json({
                "session_id": data.get("session_id", ""),
                "stopped_reason": data.get("stopped_reason", ""),
                "tokens": data.get("tokens", 0),
                "latency_ms": data.get("latency_ms", 0),
                "trace": safe_trace(trace),
            })


def render_history(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        with st.chat_message(message.get("role", "assistant")):
            if message.get("error"):
                st.error(message["error"])
            elif message.get("role") == "user":
                st.markdown(message.get("content", ""))
            else:
                render_answer(message.get("data", {}))


def main() -> None:
    st.set_page_config(page_title="IFR-QA Agent", page_icon="📄", layout="wide")
    st.title("Indian Financial-Report QA Agent")
    st.caption("Streamlit UI → FastAPI REST → LangGraph agent → page-level retrieval")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_question", "")
    st.session_state.setdefault("progress", initial_progress())

    examples = load_examples()

    def on_example_change() -> None:
        label = st.session_state.get("example_selector", "")
        for example_label, question in examples:
            if label == example_label:
                st.session_state["chat_input"] = question
                break

    backend_url = st.sidebar.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    timeout = st.sidebar.number_input(
        "Request timeout (seconds)", min_value=10, max_value=1200, value=int(REQUEST_TIMEOUT)
    )
    st.sidebar.selectbox(
        "Example questions",
        ["Custom question"] + [label for label, _ in examples],
        key="example_selector",
        on_change=on_example_change,
    )
    if st.sidebar.button("New Chat"):
        st.session_state["messages"] = []
        st.session_state["pending_question"] = ""
        st.session_state["progress"] = initial_progress()
        st.session_state["chat_input"] = ""
        st.rerun()
    ready, ready_error = backend_ready(backend_url)
    if ready:
        st.sidebar.success("Backend online")
    else:
        st.sidebar.info(f"Backend not ready: {ready_error or 'warming up'}")

    messages: list[dict[str, Any]] = st.session_state["messages"]
    pending = str(st.session_state["pending_question"])
    render_history(messages)
    if not messages and not pending:
        with st.chat_message("assistant"):
            st.markdown("Ask a question about Indian company annual reports.")

    question = st.chat_input("Ask an annual-report question...", key="chat_input",
                             disabled=bool(pending) or not ready)
    if question and question.strip() and not pending:
        add_user_message(st.session_state["messages"], question.strip())
        st.session_state["pending_question"] = question.strip()
        st.session_state["progress"] = initial_progress()
        st.rerun()

    if not pending:
        return

    progress = st.session_state["progress"]
    with st.chat_message("assistant"):
        status = st.status("Working...", expanded=True)
        answer_placeholder = st.empty()
        status_lines = ["Request received"]
        status.write("\n".join(f"- {line}" for line in status_lines))
        started = time.monotonic()
        final_data: dict[str, Any] | None = None
        error_message = ""

        def show_status(label: str) -> None:
            status_lines.append(label)
            status.write("\n".join(f"- {line}" for line in status_lines))

        try:
            for event in stream_answer(backend_url, pending, float(timeout)):
                kind = event.get("type") or event.get("event")
                if kind == "step":
                    show_status(step_status_label(event))
                elif kind == "tool":
                    show_status(tool_status_label(event))
                elif kind == "answer":
                    answer_placeholder.markdown(str(event.get("text") or ""))
                elif event.get("event") == "progress":
                    apply_progress_event(progress, event)
                elif event.get("event") == "final":
                    apply_final_event(progress, event)
                    final_data = event.get("data") or {}
                    answer_placeholder.markdown(str(final_data.get("answer") or ""))
                elif kind == "error":
                    error_message = str(event.get("message") or "Streaming error")
                    status.update(label="Failed", state="error", expanded=False)
                    break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                error_message = (
                    "API error 404: /ask/stream not found. Restart the backend with "
                    "`uv run uvicorn src.api:app --host 0.0.0.0 --port 8000`."
                )
            else:
                detail = exc.response.text[:300] if exc.response.is_closed else ""
                error_message = f"API error {exc.response.status_code}: {detail}"
        except httpx.RequestError as exc:
            error_message = f"Cannot reach FastAPI: {type(exc).__name__}: {exc}"
        except (httpx.StreamError, ValueError) as exc:
            error_message = f"Streaming interrupted: {type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        if final_data is None:
            if not error_message:
                error_message = "Streaming ended before the final answer was received."
            status.update(label=f"Failed in {elapsed:.1f} s", state="error", expanded=False)
            answer_placeholder.error(error_message)
            add_assistant_message(st.session_state["messages"], error=error_message)
        else:
            status.update(label=f"Done in {elapsed:.1f} s", state="complete", expanded=False)
            add_assistant_message(st.session_state["messages"], data=final_data)
        st.session_state["pending_question"] = ""
        st.rerun()


if __name__ == "__main__":
    main()
