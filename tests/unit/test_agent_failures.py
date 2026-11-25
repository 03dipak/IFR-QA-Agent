import json
import sqlite3

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import AgentGraph
from src.agent.llm import StubLLM


@pytest.fixture(autouse=True)
def _no_real_company_scan(monkeypatch):
    monkeypatch.setattr("src.agent.graph.detect_companies", lambda query: [])


def _accept() -> str:
    return json.dumps({"verdict": "accept", "feedback": "ok", "next_steps": []})


def test_malformed_planner_json_one_repair_then_clean_failure():
    llm = StubLLM(["not json", "still not json", _accept()])
    graph = AgentGraph(llm=llm, checkpointer=MemorySaver())
    result = graph.run("bad planner", thread_id="t-malformed")

    assert len(llm.calls) == 2
    assert sum(1 for e in result["errors"] if "planner: invalid JSON attempt" in e) == 2
    assert result["stop_reason"] == "reasoning_error"
    assert result["final_answer"]["insufficient_evidence"] is True
    assert "insufficient evidence" in result["final_answer"]["answer"].lower()
    assert result["final_answer"].get("figures", []) == []
    assert not result["final_answer"].get("citations")


def test_repeated_identical_tool_call_stops_with_reason():
    plan = json.dumps({
        "steps": [
            {"tool": "search_index", "args": {"query": "revenue", "k": 5}},
            {"tool": "search_index", "args": {"query": "revenue", "k": 5}},
            {"tool": "search_index", "args": {"query": "revenue", "k": 5}},
        ],
    })
    llm = StubLLM([plan, _accept()])
    calls = []

    def fake_search(**kwargs):
        calls.append(kwargs)
        return {"hits": [], "match_info": {}}

    graph = AgentGraph(llm=llm, tools={"search_index": fake_search}, checkpointer=MemorySaver())
    result = graph.run("revenue", thread_id="t-repeat")

    assert len(calls) == 1
    assert result["stop_reason"] not in ("", None)
    assert any(e.get("skipped") == "repeat_query" for e in result["tool_results"])


def test_step_cap_and_token_budget_exhaustion():
    step_plan = json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "1+1"}}],
    })
    revise = json.dumps({
        "verdict": "revise", "feedback": "more", "next_steps": [
            {"tool": "calculator", "args": {"expression": "2+2"}}
        ],
    })
    llm = StubLLM([step_plan, revise] + [revise for _ in range(20)] + [_accept()])
    graph = AgentGraph(llm=llm, checkpointer=MemorySaver(), max_steps=10, max_revisions=20)
    result = graph.run("compute", thread_id="t-stepcap")
    assert result["stop_reason"] == "step_cap"
    assert result.get("final_answer") is not None
    assert result["final_answer"].get("answer")

    big = "x" * 48010
    budget_llm = StubLLM([big, _accept()])
    budget_graph = AgentGraph(llm=budget_llm, checkpointer=MemorySaver(), max_tokens=12000)
    budget_result = budget_graph.run("budget", thread_id="t-budget")
    assert budget_result["stop_reason"] == "token_budget"
    assert budget_result.get("final_answer") is not None
    assert budget_result["final_answer"].get("answer")


def test_sqlite_checkpoint_resume_same_thread(tmp_path, monkeypatch):
    db_path = tmp_path / "agent_state.sqlite"
    monkeypatch.setenv("AGENT_STATE_DB", str(db_path))

    plan1 = json.dumps({"steps": [{"tool": "calculator", "args": {"expression": "1+1"}}]})
    g1 = AgentGraph(llm=StubLLM([plan1, _accept()]))
    r1 = g1.run("first question", thread_id="resume-thread")
    assert r1["final_answer"]["answer"]

    plan2 = json.dumps({"steps": [{"tool": "calculator", "args": {"expression": "2+2"}}]})
    g2 = AgentGraph(llm=StubLLM([plan2, _accept()]))
    r2 = g2.run("second question", thread_id="resume-thread")
    assert r2["final_answer"]["answer"]

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT thread_id FROM checkpoints WHERE thread_id=?", ("resume-thread",)).fetchall()
    finally:
        conn.close()
    assert rows
    history = list(g2.graph.get_state_history({"configurable": {"thread_id": "resume-thread"}}))
    queries = [s.values.get("query") for s in history if isinstance(s.values, dict)]
    assert "first question" in queries
    assert "second question" in queries


def test_no_evidence_returns_insufficient_without_invented_figure():
    plan = json.dumps({"steps": [{"tool": "search_index", "args": {"query": "revenue", "k": 5}}]})
    answer = json.dumps({
        "answer": "Revenue was 9999 crore.",
        "citations": ["missing"],
        "insufficient_evidence": False,
        "figures": [{"company_id": "x", "label": "Revenue", "period_column": "current", "value": "9999", "chunk_id": "missing"}],
    })
    llm = StubLLM([plan, answer])

    def fake_search(**kwargs):
        return {"hits": [], "match_info": {}}

    graph = AgentGraph(llm=llm, tools={"search_index": fake_search}, checkpointer=MemorySaver())
    result = graph.run("What was revenue?", thread_id="t-noevidence")

    final = result["final_answer"]
    assert final["insufficient_evidence"] is True
    assert "insufficient evidence" in final["answer"].lower()
    assert "9999" not in final["answer"]
    assert final.get("citations") == []
    assert final.get("figures") == []
