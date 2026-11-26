from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src import api
from src.agent import demo, service
from src.agents import graph as agent_graph
from src.agents.executor import Executor
from src.llmops.session_log import write_session_log
from src.llmops.tracing import Tracer
from src.ui.app import (
    add_assistant_message,
    add_user_message,
    apply_final_event,
    apply_progress_event,
    backend_ready,
    initial_progress,
    is_conversational,
    load_examples,
    parse_sse,
    progress_markdown,
)


def _result(session_id: str = "test-session") -> service.QuestionResult:
    return service.QuestionResult(
        state={
            "tool_results": [{
                "tool": "retrieve",
                "args": {"query": "Q"},
                "result": {"hits": [{
                    "chunk_id": "abcdef123456",
                    "meta": {"page": 1},
                    "text": "x",
                }]},
            }],
            "stop_reason": "critic_pass",
        },
        answer="Answer [abcdef123456 p.1]",
        citations=["[abcdef123456 p.1]"],
        raw_citations=["abcdef123456"],
        draft_answer="Answer [abcdef123456 p.1]",
        session_id=session_id,
        latency_s=0.01,
        steps=[],
        events=[],
        token_count=7,
    )


def _emit_fake_events(
    result: service.QuestionResult,
    on_event,
    *,
    calculator: bool = False,
    insufficient: bool = False,
) -> None:
    events = [
        {"event": "step", "type": "step", "node": "planner", "detail": "Planning..."},
        {"event": "step", "type": "step", "node": "executor", "detail": "Searching..."},
        {"event": "tool", "type": "tool", "name": "retrieve", "args_summary": "Q"},
    ]
    if calculator:
        events.append({"event": "tool", "type": "tool", "name": "calculator",
                       "args_summary": "1 + 1"})
    events.append({
        "event": "step", "type": "step", "node": "answer", "detail": "Writing answer...",
    })
    if not insufficient:
        events.extend([
            {"event": "step", "type": "step", "node": "critic", "detail": "Checking evidence..."},
            {"event": "step", "type": "step", "node": "answer", "detail": "Writing answer..."},
        ])
    for event in events:
        result.events.append(event)
        result.steps.append(event)
        if on_event is not None:
            on_event(event)


def _sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


def _timeline(events: list[dict]) -> dict[str, dict[str, str]]:
    timeline = initial_progress()
    for event in events:
        if event.get("event") == "final":
            apply_final_event(timeline, event)
        else:
            apply_progress_event(timeline, event)
    return timeline


def _progress(stage: str, status: str = "completed", *, tool: str = "") -> dict:
    return {"event": "progress", "stage": stage, "status": status, "tool": tool}


def test_examples_loaded_from_demo_scenarios():
    examples = load_examples()
    assert len(examples) == 5
    assert any("G-039" in label for label, _ in examples)
    assert all(question.strip() for _, question in examples)


def test_chat_history_keeps_followups_and_errors():
    messages: list[dict] = []
    add_user_message(messages, "first question")
    add_assistant_message(messages, data={"answer": "first answer"})
    add_user_message(messages, "follow-up")
    add_assistant_message(messages, error="API unavailable")
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[2]["content"] == "follow-up"
    assert messages[3]["error"] == "API unavailable"


def test_timeline_is_event_driven_without_invented_stages():
    timeline = _timeline([
        _progress("request", "running"),
        _progress("intent"),
        _progress("plan"),
        _progress("retrieval"),
        _progress("generate"),
        _progress("critique"),
        _progress("revise"),
        {"event": "final", "data": {"response_type": "financial", "answer": "x"}},
    ])
    markdown = progress_markdown(timeline)
    assert list(timeline) == ["request", "plan", "retrieval", "generate", "critique",
                              "revise", "completed"]
    assert timeline["request"]["status"] == "completed"
    assert "pending" not in markdown
    assert "Calculator execution" not in markdown
    assert "Citation" not in markdown


def test_parse_sse_handles_invalid_json():
    response = httpx.Response(
        200,
        text='data: {"event": "end"}\n\ndata: not-json\n\n',
        request=httpx.Request("POST", "http://test/ask/stream"),
    )
    events = list(parse_sse(response))
    assert events[0]["event"] == "end"
    assert events[1]["event"] == "error"


def test_ask_endpoint_keeps_contract(monkeypatch):
    def fake_run_question(question, session_id=None, on_event=None):
        return _result(session_id or "ask-session")

    monkeypatch.setattr(api, "run_question", fake_run_question)
    response = TestClient(api.app).post("/ask", json={"question": "Q"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Answer [abcdef123456 p.1]"
    assert data["evidence"][0]["pdf_page"] == 1
    assert "trace" in data and "citations" in data


def test_ask_stream_emits_real_events_and_final(monkeypatch):
    def fake_run_question(question, session_id=None, on_event=None):
        result = _result(session_id or "stream-session")
        _emit_fake_events(result, on_event)
        return result

    monkeypatch.setattr(api, "run_question", fake_run_question)
    with TestClient(api.app).stream("POST", "/ask/stream", json={"question": "Q"}) as response:
        assert response.status_code == 200
        events = _sse_events("".join(response.iter_text()))
    progress = [(e["stage"], e["status"]) for e in events if e["event"] == "progress"]
    steps = [(e.get("node"), e.get("detail")) for e in events if e.get("type") == "step"]
    tools = [e.get("name") for e in events if e.get("type") == "tool"]
    final = next(e for e in events if e["event"] == "final")
    assert ("request", "running") in progress
    assert ("completed", "completed") in progress
    assert ("planner", "Planning...") in steps
    assert ("executor", "Searching...") in steps
    assert "retrieve" in tools
    assert final["data"]["answer"] == "Answer [abcdef123456 p.1]"


@pytest.mark.parametrize("calculator,expected", [(True, True), (False, False)])
def test_ask_stream_calculation_event_follows_actual_tool(monkeypatch, calculator, expected):
    def fake_run_question(question, session_id=None, on_event=None):
        result = _result(session_id or "stream-session")
        _emit_fake_events(result, on_event, calculator=calculator)
        return result

    monkeypatch.setattr(api, "run_question", fake_run_question)
    with TestClient(api.app).stream("POST", "/ask/stream", json={"question": "Q"}) as response:
        events = _sse_events("".join(response.iter_text()))
    tools = [e.get("name") for e in events if e.get("type") == "tool"]
    assert ("calculator" in tools) is expected


def test_ask_stream_insufficient_evidence_has_no_fabricated_events(monkeypatch):
    def fake_run_question(question, session_id=None, on_event=None):
        result = _result(session_id or "stream-session")
        result.citations = []
        result.state["stop_reason"] = "citation_unresolved"
        _emit_fake_events(result, on_event, insufficient=True)
        return result

    monkeypatch.setattr(api, "run_question", fake_run_question)
    with TestClient(api.app).stream("POST", "/ask/stream", json={"question": "Q"}) as response:
        events = _sse_events("".join(response.iter_text()))
    nodes = {e.get("node") for e in events if e.get("type") == "step"}
    tools = {e.get("name") for e in events if e.get("type") == "tool"}
    assert {"planner", "executor", "answer"} <= nodes
    assert "retrieve" in tools
    assert "calculator" not in tools
    assert "critic" not in nodes


def test_ask_stream_reports_agent_error(monkeypatch):
    def failing_run_question(question, session_id=None, on_event=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "run_question", failing_run_question)
    with TestClient(api.app).stream("POST", "/ask/stream", json={"question": "Q"}) as response:
        events = _sse_events("".join(response.iter_text()))
    assert any(e["event"] == "error" and "RuntimeError" in e["message"] for e in events)
    assert events[-1]["event"] == "end"


def test_calculator_timeline_appears_only_when_tool_event_exists():
    with_calculator = _timeline([
        _progress("request", "running"),
        _progress("retrieval"),
        _progress("calculation"),
        _progress("generate"),
        {"event": "final", "data": {"response_type": "financial"}},
    ])
    without_calculator = _timeline([
        _progress("request", "running"),
        _progress("retrieval"),
        _progress("generate"),
        {"event": "final", "data": {"response_type": "financial"}},
    ])
    assert "✓ **Calculator execution**" in progress_markdown(with_calculator)
    assert "Calculator execution" not in progress_markdown(without_calculator)


def test_insufficient_evidence_timeline_has_no_invented_review_or_revision():
    timeline = _timeline([
        _progress("request", "running"),
        _progress("intent"),
        _progress("retrieval"),
        _progress("result"),
        {"event": "final", "data": {
            "response_type": "financial", "stopped_reason": "citation_unresolved",
        }},
    ])
    markdown = progress_markdown(timeline)
    assert "Document/page retrieval" in markdown
    assert "Agent execution completed" in markdown
    assert "Answer review" not in markdown
    assert "Answer revision" not in markdown
    assert "pending" not in markdown


def test_greeting_timeline_is_omitted():
    timeline = _timeline([
        _progress("request", "running"),
        _progress("completed"),
        {"event": "final", "data": {"response_type": "conversational", "answer": "Hello"}},
    ])
    assert timeline == {}
    assert progress_markdown(timeline) == ""


def test_failed_and_interrupted_requests_do_not_invent_statuses():
    failed = _timeline([
        _progress("request", "running"),
        _progress("plan", "failed"),
        {"event": "error", "message": "RuntimeError"},
    ])
    failed_markdown = progress_markdown(failed)
    assert failed["request"]["status"] == "failed"
    assert failed["plan"]["status"] == "failed"
    assert "✗ **Request failed**" in failed_markdown

    interrupted = _timeline([
        _progress("request", "running"),
        _progress("retrieval"),
    ])
    interrupted_markdown = progress_markdown(interrupted)
    assert "Request failed" not in interrupted_markdown
    assert "pending" not in interrupted_markdown
    assert "● **Request received**" in interrupted_markdown


@pytest.mark.parametrize("text,expected", [
    ("hi", "greeting"),
    ("Hello", "greeting"),
    ("Hey there!", "greeting"),
    ("Good morning", "greeting"),
    ("Thank you", "gratitude"),
    ("Thanks a lot", "gratitude"),
    ("Goodbye", "goodbye"),
    ("How are you?", "small_talk"),
    ("what can you do", "small_talk"),
])
def test_conversational_variants_are_classified(text, expected):
    assert api.classify_conversational(text) == expected
    assert api.classify_conversational("What were Adani Green total assets?") == ""
    assert api.classify_conversational(
        "For Indian Oil Corporation for FY2024 consolidated: what were their audit fees?"
    ) == ""


def test_greeting_bypasses_agent_and_logs_without_content(monkeypatch, caplog):
    def should_not_run(*args, **kwargs):
        raise AssertionError("agent must not run for greetings")

    monkeypatch.setattr(api, "run_question", should_not_run)
    with caplog.at_level(logging.INFO, logger="ifr.api"):
        response = TestClient(api.app).post("/ask", json={"question": "hi"})
    data = response.json()
    assert response.status_code == 200
    assert data["response_type"] == "conversational"
    assert data["conversation_kind"] == "greeting"
    assert data["citations"] == [] and data["evidence"] == []
    assert data["calculations"] == [] and data["trace"] == []
    assert "hi" not in caplog.text
    assert data["answer"] not in caplog.text


def test_greeting_stream_skips_financial_events(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("agent must not run for greetings")

    monkeypatch.setattr(api, "run_question", should_not_run)
    with TestClient(api.app).stream("POST", "/ask/stream", json={"question": "Thank you"}) as resp:
        events = _sse_events("".join(resp.iter_text()))
    stages = {e.get("stage") for e in events if e.get("event") == "progress"}
    final = next(e for e in events if e.get("event") == "final")
    assert stages == {"request", "completed"}
    assert final["data"]["response_type"] == "conversational"
    assert final["data"]["evidence"] == []


def test_ui_response_type_comes_from_structured_field():
    assert is_conversational({"response_type": "conversational", "answer": "What are assets?"})
    assert not is_conversational({"response_type": "financial", "answer": "hi"})
    assert not is_conversational({"answer": "hi"})


def test_executor_trace_omits_arguments_and_output():
    class CaptureTracer:
        def __init__(self):
            self.rows = []

        def record(self, stage, **kwargs):
            self.rows.append({"stage": stage, **kwargs})

    tracer = CaptureTracer()
    executor = Executor(tools={"retrieve": lambda query: ["chunk"]}, tracer=tracer)
    step = SimpleNamespace(tool="retrieve", args={"query": "secret question"},
                           sub_question="secret subquestion")
    executor.run([step])
    dumped = json.dumps(tracer.rows)
    assert "secret" not in dumped
    assert "chunk" not in dumped
    assert tracer.rows[0]["tool"] == "retrieve"


def test_run_agent_does_not_write_chat_session_file(tmp_path, monkeypatch):
    class FakeGraph:
        def invoke(self, state):
            return {**state, "answer": "secret answer", "citations": [], "steps": [],
                    "stopped_reason": "done", "plan_completions": [], "draft_completions": [],
                    "revision_completions": [], "critique_completion": None}

    monkeypatch.setattr(agent_graph, "build_agent_graph", FakeGraph)
    monkeypatch.chdir(tmp_path)
    tracer = Tracer("privacy", root=tmp_path)
    agent_graph.run_agent("secret question", session_id="privacy", client=object(),
                          executor=object(), tracer=tracer)
    assert not (tmp_path / "data" / "sessions").exists()


def test_stream_logging_omits_question_and_answer(monkeypatch, caplog):
    def fake_run_question(question, session_id=None, on_event=None):
        result = _result(session_id or "log-session")
        result.answer = "secret answer"
        return result

    monkeypatch.setattr(api, "run_question", fake_run_question)
    with caplog.at_level(logging.INFO, logger="ifr.api"):
        with TestClient(api.app).stream(
            "POST", "/ask/stream", json={"question": "secret question"}
        ) as response:
            "".join(response.iter_text())
    assert "ask_stream" in caplog.text
    assert "secret question" not in caplog.text
    assert "secret answer" not in caplog.text


@pytest.mark.skipif(
    not Path(os.environ.get("AGENT_INDEX_PATH", "results/full_index_v2/bge_small")).exists(),
    reason="AGENT_INDEX_PATH index unavailable",
)
def test_health_and_ask_smoke_with_current_graph(monkeypatch):
    def fake_run_question(question, session_id=None, on_event=None):
        result = _result(session_id or "smoke-session")
        result.answer = "coforge Revenue from operations: 91,790 [abcdef123456 p.1]"
        return result

    monkeypatch.setattr(api, "run_question", fake_run_question)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/ask", json={"question": "What is revenue?"})
    assert response.status_code == 200
    data = response.json()
    assert "91,790" in data["answer"]
    assert data["citations"] == ["[abcdef123456 p.1]"]
    assert data["evidence"][0]["pdf_page"] == 1
    assert data["citation_status"] == "resolved"


def test_ready_waits_for_warmup(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_warm_index():
        started.set()
        release.wait(2)
        return "numpy"

    monkeypatch.setattr(api, "_INDEX_HEALTHY", True)
    monkeypatch.setattr(api, "READY", False)
    monkeypatch.setattr(api, "READY_ERROR", "")
    monkeypatch.setattr(api, "INDEX_BACKEND", "unknown")
    monkeypatch.setattr(api, "warm_index", slow_warm_index)
    monkeypatch.setattr(api, "warm_meta_rows", lambda: 1)
    monkeypatch.setattr(api, "warm_reranker", lambda: True)
    monkeypatch.setattr(api, "warm_query_embedding", lambda: 0.0)
    with TestClient(api.app) as client:
        assert started.wait(2)
        assert client.get("/ready").json()["ready"] is False
        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if client.get("/ready").json()["ready"]:
                break
            time.sleep(0.01)
        assert client.get("/ready").json() == {
            "ready": True, "backend": "numpy",
            "stages": ["index", "meta_rows", "reranker", "query_embedding"],
        }


def test_health_ok_but_not_ready_when_index_missing(monkeypatch):
    def fail_warm_index():
        raise FileNotFoundError("index path missing")

    monkeypatch.setattr(api, "_INDEX_HEALTHY", False)
    monkeypatch.setattr(api, "READY", False)
    monkeypatch.setattr(api, "READY_ERROR", "")
    monkeypatch.setattr(api, "warm_index", fail_warm_index)
    monkeypatch.setattr(api, "warm_query_embedding", lambda: 0.0)
    with TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        ready = client.get("/ready").json()
        assert ready["ready"] is False
        assert "index path missing" in ready["error"]


def test_ui_backend_ready_uses_ready_endpoint(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ready": True, "backend": "numpy", "error": ""}

    monkeypatch.setattr(
        "src.ui.app.httpx.get",
        lambda url, timeout: FakeResponse(),
    )
    assert backend_ready("http://health-test") == (True, "")


def test_session_log_writer_appends_one_jsonl_line(tmp_path):
    root = tmp_path / "sessions"
    steps = [{"node": "planner", "tool": "", "args_summary": ""}]
    write_session_log(
        "what is profit?", "604 (Rs. in crore)", ["[abcdef123456 p.1]"], 1.25,
        steps, session_id="log-writer", root=root,
    )
    lines = (root / "log-writer.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["ts"][:10] == time.strftime("%Y-%m-%d")
    assert row["question"] == "what is profit?"
    assert row["answer"] == "604 (Rs. in crore)"
    assert row["citations"] == ["[abcdef123456 p.1]"]
    assert row["latency_s"] == 1.25
    assert row["steps"] == steps
    assert row["error"] == ""


def test_cli_api_parity_for_t3_gold_and_aurobindo(tmp_path, monkeypatch, capsys):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    g001 = (
        "Between Deepak Nitrite and Srf, on total assets for the year ended 31 March 2025 "
        "consolidated results, which prints the larger figure, and what are the two figures "
        "(in Rs. in crore, as printed on each statement)?"
    )
    g004 = (
        "Compare Maruti Suzuki and Bosch on total assets for the year ended 31 March 2025 "
        "consolidated results (both Maruti Suzuki and Bosch at the same basis): who reports "
        "more, and by how much, in Rs. in million as printed?"
    )
    aurobindo = (
        "Compare Aurobindo Pharma and Dr Reddys on total assets for the year ended 31 March "
        "2025 consolidated results (both Aurobindo Pharma and Dr Reddys at the same basis): "
        "who reports more, and by how much, in Rs. in million as printed?"
    )
    questions = [
        ("t3_revenue", "For t3, what was Revenue from operations for FY2025 consolidated?"),
        ("G-001", g001),
        ("G-004", g004),
        ("G-010", aurobindo),
    ]

    class FakeGraph:
        def __init__(self, llm):
            self.budget = SimpleNamespace(used=7)

        def stream(self, question, thread_id="default", *, callbacks=None, metadata=None):
            final = {
                "answer": f"answer::{question}",
                "citations": ["abcdef123456"],
                "figures": [{"chunk_id": "abcdef123456", "page": 1}],
            }
            for node in ("planner", "executor", "critic", "answer"):
                yield {"event": "step", "type": "step", "node": node, "detail": node}
            yield {
                "event": "tool", "type": "tool", "name": "search_index",
                "args_summary": "query",
            }
            if "aurobindo" in question.lower():
                yield {
                    "event": "tool", "type": "tool", "name": "search_index",
                    "args_summary": "query",
                }
            yield {"event": "state", "type": "state", "state": {
                "plan": [], "tool_results": [{
                    "tool": "search_index", "args": {"query": "query"},
                    "result": {"hits": [{"chunk_id": "abcdef123456",
                                         "meta": {"page": 1}}]},
                }],
                "draft_answer": f"answer::{question}", "critique": {}, "errors": [],
                "steps_used": 1, "stop_reason": "accepted", "token_count": 7,
                "final_answer": final,
            }}

    def parse_cli(output: str) -> tuple[str, list[str]]:
        answer = output.split("Final answer:\n", 1)[1].split("\nSynthesis:\n", 1)[0]
        citations = output.split("Citations:\n", 1)[1].split("\nInsufficient evidence:", 1)[0]
        return answer, json.loads(citations)

    monkeypatch.setenv("LLM_CHAT_URL", "http://test")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SESSION_LOG_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(service, "EnvLLM", lambda: object())
    monkeypatch.setattr(service, "AgentGraph", FakeGraph)
    monkeypatch.setattr(demo, "warm_index", lambda: None)

    real_run_question = demo.run_question
    captured_results: dict[str, service.QuestionResult] = {}
    client = TestClient(api.app)
    for label, question in questions:
        session_id = f"parity-cli-{label}"

        def cli_run_question(question, session_id=None, on_event=None):
            result = real_run_question(question, session_id=session_id, on_event=on_event)
            captured_results[label] = result
            return result

        monkeypatch.setattr(demo, "run_question", cli_run_question)
        capsys.readouterr()
        assert demo.main([question, "--thread-id", session_id]) == 0
        cli_answer, cli_citations = parse_cli(capsys.readouterr().out)
        response = client.post(
            "/ask", json={"question": question, "session_id": f"parity-api-{label}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert cli_answer == data["answer"]
        assert cli_citations == data["citations"]
        assert captured_results[label].answer == data["answer"]
        assert captured_results[label].citations == data["citations"]

    aurobindo_result = captured_results["G-010"]
    for event in aurobindo_result.events:
        if event["type"] == "step":
            print(f"{event['node']}={event['elapsed_s']}s")
    calls = sum(
        1 for event in aurobindo_result.events
        if event["type"] == "tool" and event["name"] == "search_index"
    )
    print(f"search_index_calls={calls}")
    captured = capsys.readouterr().out
    assert "planner=" in captured
    assert "answer=" in captured
    assert "search_index_calls=2" in captured

    rows = [
        json.loads(line)
        for path in (tmp_path / "sessions").glob("parity-cli-*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2 * len(questions)
    assert [row["status"] for row in rows].count("started") == len(questions)
    assert [row["status"] for row in rows].count("ok") == len(questions)
    assert all(all("type" in step for step in row["steps"]) for row in rows)


def test_run_question_without_langfuse_keys_disables_tracing(
    monkeypatch, tmp_path, caplog
):
    class FakeGraph:
        def __init__(self, llm):
            self.budget = SimpleNamespace(used=1)

        def stream(self, question, thread_id="default", *, callbacks=None, metadata=None):
            assert callbacks is None
            yield {"event": "state", "type": "state", "state": {
                "final_answer": {"answer": "ok", "citations": []},
                "stop_reason": "accepted",
                "token_count": 1,
            }}

    def forbidden(*args, **kwargs):
        raise AssertionError("Langfuse must not be initialized without keys")

    monkeypatch.setenv("LLM_CHAT_URL", "http://test")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SESSION_LOG_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.setattr(service, "Langfuse", forbidden)
    monkeypatch.setattr(service, "CallbackHandler", forbidden)
    monkeypatch.setattr(service, "EnvLLM", lambda: object())
    monkeypatch.setattr(service, "AgentGraph", FakeGraph)
    monkeypatch.setattr(service, "_LANGFUSE_WARNINGS", set())

    with caplog.at_level(logging.WARNING, logger="ifr.service"):
        result = service.run_question("question", session_id="no-trace-session")

    assert result.answer == "ok"
    log_rows = [
        json.loads(line)
        for line in (tmp_path / "sessions" / "no-trace-session.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in log_rows] == ["started", "ok"]
    warnings = [
        record.message for record in caplog.records
        if "Langfuse tracing disabled" in record.message
    ]
    assert warnings == ["Langfuse tracing disabled: missing configuration"]


def test_run_question_attaches_langfuse_handler_and_final_metadata(
    monkeypatch, tmp_path
):
    captured: dict[str, Any] = {"updates": [], "flushes": 0}
    handler_instances: list[Any] = []

    class FakeSpan:
        def update(self, **kwargs):
            captured["updates"].append(kwargs)

    class FakeSpanContext:
        def __enter__(self):
            return FakeSpan()

        def __exit__(self, exc_type, exc, tb):
            captured["exited"] = True
            return False

    class FakeClient:
        def start_as_current_observation(self, **kwargs):
            captured.setdefault("starts", []).append(kwargs)
            return FakeSpanContext()

        def flush(self):
            captured["flushes"] += 1

    class FakeHandler:
        def __init__(self, **kwargs):
            captured["handler_kwargs"] = kwargs
            handler_instances.append(self)

    class FakeGraph:
        def __init__(self, llm):
            self.budget = SimpleNamespace(used=3)

        def stream(self, question, thread_id="default", *, callbacks=None, metadata=None):
            captured["callbacks"] = callbacks
            captured["metadata"] = metadata
            yield {
                "event": "tool", "type": "tool", "name": "search_index",
                "args_summary": "total assets",
            }
            yield {"event": "state", "type": "state", "state": {
                "final_answer": {"answer": "ok", "citations": []},
                "stop_reason": "accepted",
                "token_count": 3,
            }}

    monkeypatch.setenv("LLM_CHAT_URL", "http://test")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_SESSION_LOG_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://langfuse.test")
    monkeypatch.setattr(service, "Langfuse", lambda **kwargs: FakeClient())
    monkeypatch.setattr(service, "CallbackHandler", FakeHandler)
    monkeypatch.setattr(service, "EnvLLM", lambda: object())
    monkeypatch.setattr(service, "AgentGraph", FakeGraph)
    monkeypatch.setattr(service, "_LANGFUSE_CLIENTS", {})

    result = service.run_question("question", session_id="trace-session")

    assert result.answer == "ok"
    assert captured["starts"][0]["name"] == "question"
    assert captured["starts"][0]["as_type"] == "chain"
    assert [
        kwargs for kwargs in captured["starts"] if kwargs.get("as_type") == "tool"
    ] == [{
        "name": "search_index",
        "as_type": "tool",
        "input": {"name": "search_index", "args_summary": "total assets"},
    }]
    assert captured["handler_kwargs"] == {"public_key": "pk-test"}
    assert captured["callbacks"] == handler_instances
    assert captured["metadata"]["langfuse_session_id"] == "trace-session"
    assert captured["metadata"]["session_id"] == "trace-session"
    assert captured["exited"] is True
    assert captured["flushes"] == 1
    update = captured["updates"][-1]
    assert update["metadata"]["session_id"] == "trace-session"
    assert update["metadata"]["stop_reason"] == "accepted"
    assert update["metadata"]["error_type"] == "other"
    assert update["metadata"]["token_count"] == 3
    assert isinstance(update["metadata"]["latency_s"], float)
    assert update["output"] == {"answer": "ok", "citations": []}
    log_rows = [
        json.loads(line)
        for line in (tmp_path / "sessions" / "trace-session.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in log_rows] == ["started", "ok"]
