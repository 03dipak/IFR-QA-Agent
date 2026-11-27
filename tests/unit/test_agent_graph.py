import json
from pathlib import Path

import numpy as np
import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from src.agent import tools
from src.agent.graph import AgentGraph, _enforce_plan_tools, _growth_requested
from src.agent.llm import StubLLM
from src.agent.schemas import Critique, Plan


@pytest.fixture(autouse=True)
def _no_real_company_scan(monkeypatch):
    monkeypatch.setattr("src.agent.graph.detect_companies", lambda query: [])


def _accept() -> str:
    return json.dumps({"verdict": "accept", "feedback": "ok", "next_steps": []})


def _revise() -> str:
    return json.dumps({
        "verdict": "revise",
        "feedback": "retry",
        "next_steps": [{"tool": "calculator", "args": {"expression": "1+1"}}],
    })


class _FakeIndex:
    def __init__(self, rows: list[dict], search_rows: list[dict] | None = None):
        self.rows = rows
        self.ids = [row["chunk_id"] for row in rows]
        self.chunk_metas = [row.get("meta") or {} for row in rows]
        self.embeddings = np.asarray(
            [row.get("embedding", [1.0, 0.0, 0.0, 0.0]) for row in rows],
            dtype=np.float32,
        )
        self.meta = {"dim": self.embeddings.shape[1]}
        self.embedder = "hash384"
        self.search_rows = search_rows if search_rows is not None else rows
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[dict]:
        self.search_calls.append((query, k))
        return [dict(row) for row in self.search_rows[:k]]

    def _chunk_at(self, position: int) -> dict:
        row = dict(self.rows[position])
        row.pop("embedding", None)
        return row


def _fake_index(monkeypatch, index: _FakeIndex, vector: list[float] | None = None) -> None:
    monkeypatch.setattr(tools, "_open_index", lambda path: index)
    vector_array = np.asarray(vector or [1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(tools, "_query_embedding", lambda query, idx, path: vector_array)


def test_graph_completes_end_to_end():
    planner = json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "2+2"}}],
    })
    llm = StubLLM([planner, _accept()])
    graph = AgentGraph(llm=llm, checkpointer=MemorySaver())
    result = graph.run("What is 2+2?", thread_id="t1")

    assert result["steps_used"] == 1
    assert result["iteration"] == 0
    assert result["critique"]["verdict"] == "accept"
    assert "4" in result["draft_answer"]
    assert result["errors"] == []
    assert len(llm.calls) == 3


def test_revision_cap_is_respected():
    planner = json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "1+1"}}],
    })
    llm = StubLLM([planner, _revise(), _revise(), _revise(), _revise()])
    calls = {"count": 0}

    def counted_calculator(expression: str) -> int:
        calls["count"] += 1
        return 2

    graph = AgentGraph(
        llm=llm,
        tools={"calculator": counted_calculator},
        checkpointer=MemorySaver(),
        max_revisions=3,
    )
    result = graph.run("Count revisions", thread_id="t2")

    assert calls["count"] == 4
    assert result["iteration"] == 3
    assert result["steps_used"] == 4
    assert result["critique"]["verdict"] == "accept"
    assert len(llm.calls) == 6


def test_malformed_json_is_retried_then_falls_back():
    critic = _accept()
    llm = StubLLM(["not json", "still not json", "{bad", critic])

    def fake_search(query: str, k: int = 5):
        return [{"chunk_id": "c1", "text": f"{query}:{k}"}]

    graph = AgentGraph(
        llm=llm,
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("fallback query", thread_id="t3")

    assert len(llm.calls) == 2
    assert sum(1 for e in result["errors"] if "planner: invalid JSON attempt" in e) == 2
    assert any("planner: fallback used" in e for e in result["errors"])
    assert result["stop_reason"] == "reasoning_error"
    assert result["final_answer"]["insufficient_evidence"] is True


def test_sqlite_checkpointer_persists_thread_history(tmp_path, monkeypatch):
    db_path = tmp_path / "agent_state.sqlite"
    monkeypatch.setenv("AGENT_STATE_DB", str(db_path))

    first_plan = json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "1+1"}}],
    })
    first_graph = AgentGraph(llm=StubLLM([first_plan, _accept()]))
    first_graph.run("first-turn query", thread_id="persistent-thread")

    second_plan = json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "2+2"}}],
    })
    second_graph = AgentGraph(llm=StubLLM([second_plan, _accept()]))
    second_graph.run("second-turn query", thread_id="persistent-thread")

    history = list(
        second_graph.graph.get_state_history(
            {"configurable": {"thread_id": "persistent-thread"}}
        )
    )
    assert db_path.is_file()
    assert any(state.values.get("query") == "first-turn query" for state in history)


def test_tool_exception_does_not_crash_run():
    planner = json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "1/0"}}],
    })
    llm = StubLLM([planner, _accept()])

    def broken_calculator(expression: str) -> int:
        raise ValueError(f"bad tool input: {expression}")

    graph = AgentGraph(
        llm=llm,
        tools={"calculator": broken_calculator},
        checkpointer=MemorySaver(),
    )
    result = graph.run("trigger tool failure", thread_id="t4")

    assert result["steps_used"] == 1
    assert any("tool calculator failed" in error for error in result["errors"])
    assert result["critique"]["verdict"] == "accept"
    assert "No tool results available" not in result["draft_answer"]


def test_unknown_tool_is_rejected_without_step():
    graph = AgentGraph(llm=StubLLM([]), checkpointer=MemorySaver())
    state = {
        "query": "unknown tool",
        "plan": [{"tool": "text_parser", "args": {"query": "x"}}],
        "tool_results": [],
        "steps_used": 0,
        "errors": [],
    }
    result = graph._executor(state)

    assert result["steps_used"] == 0
    assert any("unknown_tool" in err for err in result["errors"])
    assert result["tool_results"][0]["error"] == "unknown_tool"
    assert result["tool_results"][0]["skipped"] == "unknown_tool"


def test_two_unknown_tools_go_straight_to_synthesis():
    graph = AgentGraph(llm=StubLLM([]), checkpointer=MemorySaver())
    state = {
        "query": "unknown tools",
        "plan": [
            {"tool": "text_parser", "args": {"query": "x"}},
            {"tool": "data_cleaner", "args": {"query": "y"}},
            {"tool": "calculator", "args": {"expression": "1+1"}},
        ],
        "tool_results": [],
        "steps_used": 0,
        "errors": [],
    }
    result = graph._executor(state)

    assert result["steps_used"] == 0
    assert result["stop_reason"] == "accepted"
    assert len(result["tool_results"]) == 2
    assert "executor: two unknown tools triggered synthesis" in result["errors"]


def test_schema_tool_whitelist():
    with pytest.raises(ValidationError):
        Plan.model_validate({"steps": [{"tool": "text_parser", "args": {}}]})
    with pytest.raises(ValidationError):
        Critique.model_validate({
            "verdict": "accept",
            "next_steps": [{"tool": "text_parser", "args": {}}],
        })
    assert Plan.model_validate({"steps": [{"tool": "calculator", "args": {}}]})
    assert Critique.model_validate({"verdict": "accept", "next_steps": []})


def test_repeat_search_is_blocked_and_routes_to_extraction():
    planner1 = json.dumps({
        "steps": [
            {"tool": "search_index", "args": {"query": "adani revenue", "k": 5}},
            {"tool": "search_index", "args": {"query": "adani revenue", "k": 5}},
        ],
    })
    planner2 = json.dumps({
        "steps": [
            {"tool": "search_index", "args": {"query": "adani revenue FY2023", "k": 5}},
        ],
    })
    llm = StubLLM([planner1, planner2, _accept()])
    calls: list[tuple[str, int]] = []

    def fake_search(query: str, k: int = 5, company: str | None = None, **kwargs):
        calls.append((query, k))
        return [{"chunk_id": f"id-{len(calls)}", "meta": {"company_id": "adani"}}]

    graph = AgentGraph(
        llm=llm,
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("adani revenue", thread_id="t6")

    assert calls == [("Revenue from operations", 20)]
    assert any("repeat search blocked" in err for err in result["errors"])
    assert any(entry.get("skipped") for entry in result["tool_results"])
    assert result["stop_reason"] == "accepted"


def test_stop_reason_accepted():
    planner = json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "1+1"}}],
    })
    graph = AgentGraph(llm=StubLLM([planner, _accept()]), checkpointer=MemorySaver())
    result = graph.run("calc", thread_id="t7")
    assert result["stop_reason"] == "accepted"


def test_stop_reason_revision_cap():
    planner = json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "1+1"}}],
    })
    llm = StubLLM([planner, _revise(), _revise(), _revise(), _revise()])
    graph = AgentGraph(llm=StubLLM(llm._responses) if False else llm, checkpointer=MemorySaver(), max_revisions=3)
    result = graph.run("revise", thread_id="t8")
    assert result["stop_reason"] == "revision_cap"


def test_stop_reason_step_cap():
    planner = json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "1+1"}}],
    })
    llm = StubLLM([planner, _revise()])
    graph = AgentGraph(
        llm=llm,
        checkpointer=MemorySaver(),
        max_steps=1,
        max_revisions=3,
    )
    result = graph.run("step cap", thread_id="t9")
    assert result["stop_reason"] == "step_cap"


def test_stop_reason_token_budget():
    llm = StubLLM([json.dumps({"steps": [{"tool": "calculator", "args": {"expression": "1"}}]})])
    graph = AgentGraph(llm=llm, checkpointer=MemorySaver(), max_tokens=0)
    result = graph.run("budget", thread_id="t10")
    assert result["stop_reason"] == "token_budget"


def test_per_company_search_split():
    planner = json.dumps({
        "steps": [
            {
                "tool": "search_index",
                "args": {"query": "Adani revenue 2023", "k": 5, "company": "adani"},
            },
            {
                "tool": "search_index",
                "args": {"query": "TCS revenue 2023", "k": 5, "company": "tcs"},
            },
        ],
    })
    calls: list[dict] = []

    def fake_search(query: str, k: int = 5, company: str | None = None, **kwargs):
        calls.append({"query": query, "k": k, "company": company})
        return [{"chunk_id": f"{company}-1", "meta": {"company_id": company}}]

    graph = AgentGraph(
        llm=StubLLM([planner, _accept()]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("Compare Adani and TCS revenue 2023", thread_id="t11")

    assert len(calls) == 2
    assert {c["company"] for c in calls} == {"adani", "tcs"}
    assert all(c["company"] for c in calls)
    assert all(c["query"] == "Revenue from operations" for c in calls)
    assert all(c["k"] >= 20 for c in calls)
    assert result["stop_reason"] == "accepted"


def test_fenced_json_parses():
    planner = (
        "```json\n"
        + json.dumps({
            "steps": [{"tool": "calculator", "args": {"expression": "2+2"}}],
        })
        + "\n```"
    )
    graph = AgentGraph(llm=StubLLM([planner, _accept()]), checkpointer=MemorySaver())
    result = graph.run("fenced", thread_id="t12")

    assert result["plan"][0]["tool"] == "calculator"
    assert "4" in result["draft_answer"]
    assert result["errors"] == []


def test_raw_output_logged_on_parse_failure():
    bad = "NO JSON {broken"
    llm = StubLLM([bad, bad, bad, _accept()])

    def fake_search(query: str, k: int = 5, company: str | None = None, **kwargs):
        return []

    graph = AgentGraph(
        llm=llm,
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("bad json", thread_id="t13")

    assert any(f"raw_output={bad}" in error for error in result["errors"])
    assert result["stop_reason"] == "reasoning_error"
    assert result["final_answer"]["insufficient_evidence"] is True


def test_exact_company_filter_excludes_substring(monkeypatch):
    rows = [
        {"chunk_id": "sub", "meta": {"company_id": "tcs_group"}},
        {"chunk_id": "exact", "meta": {"company_id": "TCS"}},
    ]
    _fake_index(monkeypatch, _FakeIndex(rows))
    res = tools.search_index("TCS revenue", k=5, company="tcs")
    hits = res["hits"]

    assert [hit["chunk_id"] for hit in hits] == ["exact"]
    assert hits[0]["match_mode"] == "exact"
    assert res["match_info"]["filter_mode"] == "masked"


def test_filter_first_returns_k_hits_when_post_filter_returned_one(monkeypatch):
    rows = [
        {
            "chunk_id": "miss-1",
            "meta": {"company_id": "deepak_nitrite", "period": "FY2025"},
            "embedding": [0.0, 1.0, 0.0, 0.0],
        },
        {
            "chunk_id": "hit-1",
            "meta": {"company_id": "srf", "period": "FY2025"},
            "embedding": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "chunk_id": "miss-2",
            "meta": {"company_id": "deepak_nitrite", "period": "FY2025"},
            "embedding": [0.0, 1.0, 0.0, 0.0],
        },
        {
            "chunk_id": "hit-2",
            "meta": {"company_id": "srf", "period": "FY2025"},
            "embedding": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "chunk_id": "hit-3",
            "meta": {"company_id": "srf", "period": "FY2025"},
            "embedding": [1.0, 0.0, 0.0, 0.0],
        },
    ]
    index = _FakeIndex(rows, search_rows=rows[:3])
    _fake_index(monkeypatch, index)

    res = tools.search_index("revenue", k=3, company="srf", period="FY2025")

    assert [hit["chunk_id"] for hit in res["hits"]] == ["hit-1", "hit-2", "hit-3"]
    assert all(hit["meta"]["company_id"] == "srf" for hit in res["hits"])
    assert res["match_info"]["masked_rows"] == 3
    assert index.search_calls == []


def test_detect_companies_adani_vs_tcs():
    assert tools.detect_companies(
        "adani vs tcs", ids=["adani_green", "tcs", "tata_steel"]
    ) == ["adani_green", "tcs"]


def test_executor_forces_detected_companies(monkeypatch):
    monkeypatch.setattr(
        "src.agent.graph.detect_companies",
        lambda query: ["adani_green", "tcs"],
    )
    planner = json.dumps({
        "steps": [
            {"tool": "search_index", "args": {"query": "Adani revenue", "k": 5}},
            {"tool": "search_index", "args": {"query": "TCS revenue", "k": 5}},
        ],
    })
    calls: list[str | None] = []

    def fake_search(
        query: str, k: int = 5, company: str | None = None, **kwargs,
    ) -> list[dict]:
        calls.append(company)
        return []

    graph = AgentGraph(
        llm=StubLLM([planner, _accept()]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    graph.run("adani vs tcs", thread_id="t14")

    assert calls == ["adani_green", "tcs"]


def _accept_plan() -> str:
    return json.dumps({
        "steps": [{"tool": "calculator", "args": {"expression": "1+1"}}],
    })


def test_critic_string_next_steps_are_coerced(monkeypatch):
    monkeypatch.setattr(
        "src.agent.graph.detect_companies", lambda query: ["tcs"]
    )
    critic = json.dumps({
        "verdict": "accept",
        "feedback": "ok",
        "next_steps": ["TCS revenue 2023", "   "],
    })
    graph = AgentGraph(
        llm=StubLLM([_accept_plan(), critic]),
        checkpointer=MemorySaver(),
    )
    result = graph.run("tcs", thread_id="t15")

    assert result["critique"]["next_steps"] == [{
        "tool": "search_index",
        "args": {"query": "TCS revenue 2023", "k": 5, "company": "tcs"},
    }]
    assert result["errors"] == []


def test_critic_null_next_steps_becomes_empty_list():
    critic = json.dumps({"verdict": "accept", "feedback": "ok", "next_steps": None})
    graph = AgentGraph(
        llm=StubLLM([_accept_plan(), critic]),
        checkpointer=MemorySaver(),
    )
    result = graph.run("calc", thread_id="t16")

    assert result["critique"]["next_steps"] == []
    assert result["errors"] == []


def test_critic_object_next_steps_unchanged():
    steps = [{"tool": "calculator", "args": {"expression": "2+2"}}]
    critic = json.dumps({
        "verdict": "accept",
        "feedback": "ok",
        "next_steps": steps,
    })
    graph = AgentGraph(
        llm=StubLLM([_accept_plan(), critic]),
        checkpointer=MemorySaver(),
    )
    result = graph.run("calc", thread_id="t17")

    assert result["critique"]["next_steps"] == steps
    assert result["errors"] == []


def test_synthesis_returns_citations():
    planner = json.dumps({
        "steps": [{"tool": "search_index", "args": {"query": "revenue", "k": 5}}],
    })
    answer = json.dumps({
        "answer": "Revenue was 100.",
        "citations": ["c1"],
        "insufficient_evidence": False,
    })

    def fake_search(query: str, k: int = 5, **kwargs):
        return {"hits": [{"chunk_id": "c1", "text": "Revenue was 100 in FY2024."}]}

    graph = AgentGraph(
        llm=StubLLM([planner, _accept(), answer]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("revenue", thread_id="t18")

    assert result["final_answer"]["citations"] == ["c1"]
    assert result["final_answer"]["insufficient_evidence"] is False
    assert not result["final_answer"].get("flagged_ungrounded")


def test_ungrounded_figure_is_flagged():
    planner = json.dumps({
        "steps": [{"tool": "search_index", "args": {"query": "revenue", "k": 5}}],
    })
    answer = json.dumps({
        "answer": "Revenue was 999.",
        "citations": ["c1"],
        "insufficient_evidence": False,
    })

    def fake_search(query: str, k: int = 5, **kwargs):
        return {"hits": [{"chunk_id": "c1", "text": "Revenue was 100 in FY2024."}]}

    graph = AgentGraph(
        llm=StubLLM([planner, _accept(), answer]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("revenue", thread_id="t19")

    assert result["final_answer"].get("flagged_ungrounded") is True
    assert any("ungrounded_figure: 999" in e for e in result["errors"])


def test_insufficient_evidence_passes_through():
    planner = json.dumps({
        "steps": [{"tool": "search_index", "args": {"query": "revenue", "k": 5}}],
    })
    answer = json.dumps({
        "answer": "Insufficient evidence in retrieved chunks.",
        "citations": [],
        "insufficient_evidence": True,
    })

    def fake_search(query: str, k: int = 5, **kwargs):
        return {"hits": []}

    graph = AgentGraph(
        llm=StubLLM([planner, _accept(), answer]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("revenue", thread_id="t20")

    assert result["final_answer"]["insufficient_evidence"] is True


def test_period_filter_excludes_empty_period_rows(monkeypatch):
    rows = [
        {"chunk_id": "a", "meta": {"period": ""}, "text": "A"},
        {"chunk_id": "b", "meta": {"period": "FY2024"}, "text": "B"},
        {"chunk_id": "c", "meta": {}, "text": "C"},
    ]
    _fake_index(monkeypatch, _FakeIndex(rows))
    res = tools.search_index("revenue", k=5, period="FY2024")

    assert [h["chunk_id"] for h in res["hits"]] == ["b"]
    assert res["match_info"]["excluded_empty_period"] == 2


def test_relaxation_order(monkeypatch):
    rows = [{
        "chunk_id": "x1",
        "meta": {
            "company_id": "srf",
            "period": "FY2025",
            "statement_kind": "income_statement",
        },
        "text": "Revenue 100",
    }]
    _fake_index(monkeypatch, _FakeIndex(rows))
    res = tools.search_index(
        "q", k=5, company="srf", period="FY2025", statement_kind="balance_sheet"
    )

    assert res["match_info"]["relaxed"] == ["drop_statement_kind"]
    assert [h["chunk_id"] for h in res["hits"]] == ["x1"]


def test_repeat_blocked_does_not_consume_step_or_loop():
    planner = json.dumps({
        "steps": [
            {"tool": "search_index", "args": {"query": "q1", "k": 5}},
            {"tool": "search_index", "args": {"query": "q1", "k": 5}},
            {"tool": "search_index", "args": {"query": "q1", "k": 5}},
        ]
    })
    calls = {"n": 0}

    def fake_search(query: str, k: int = 5, **kwargs):
        calls["n"] += 1
        return {"hits": [{"chunk_id": "c1", "text": "x"}], "match_info": {}}

    graph = AgentGraph(
        llm=StubLLM([planner, _accept()]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("q", thread_id="t21")

    assert calls["n"] == 1
    assert result["steps_used"] == 1
    assert any("consecutive repeat guard" in e for e in result["errors"])
    assert result["stop_reason"] == "accepted"


def test_grounding_skipped_on_insufficient_evidence():
    planner = json.dumps({
        "steps": [{"tool": "search_index", "args": {"query": "revenue", "k": 5}}],
    })
    answer = json.dumps({
        "answer": "Revenue was 999 in 2024.",
        "citations": [],
        "insufficient_evidence": True,
    })

    def fake_search(query: str, k: int = 5, **kwargs):
        return {"hits": [{"chunk_id": "c1", "text": "no numbers"}]}

    graph = AgentGraph(
        llm=StubLLM([planner, _accept(), answer]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
        max_revisions=0,
    )
    result = graph.run("revenue 2024", thread_id="t22")

    assert result["final_answer"]["insufficient_evidence"] is True
    assert not result["final_answer"].get("flagged_ungrounded")
    assert not any("ungrounded_figure" in e for e in result["errors"])


def test_repeat_key_differs_by_company():
    planner = json.dumps({
        "steps": [
            {"tool": "search_index", "args": {"query": "revenue", "k": 5,
                                              "company": "srf"}},
            {"tool": "search_index", "args": {"query": "revenue", "k": 5,
                                              "company": "deepak_nitrite"}},
        ],
    })
    calls: list[tuple[str, int, str | None]] = []

    def fake_search(query: str, k: int = 5, company: str | None = None, **kwargs) -> dict:
        calls.append((query, k, company))
        return {"hits": [], "match_info": {}}

    graph = AgentGraph(
        llm=StubLLM([planner, _accept()]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    graph.run("compare companies", thread_id="t-repeat-key")

    assert calls == [("revenue", 20, "srf"), ("revenue", 20, "deepak_nitrite")]


def test_alias_resolution_maps_names_to_index_ids():
    ids = ["aurobindo_pharma", "deepak_nitrite", "dr_reddys", "srf", "tcs"]
    detected = tools.detect_companies(
        "Compare Aurobindo, Dr Reddy's and Deepak Nitrite with SRF", ids=ids
    )

    assert detected == ["aurobindo_pharma", "deepak_nitrite", "dr_reddys", "srf"]
    unresolved: list[str] = []
    assert tools.detect_companies("Zomato revenue", ids=ids, unresolved=unresolved) == []
    assert "zomato" in unresolved


def test_company_mismatch_is_flagged(monkeypatch):
    monkeypatch.setattr("src.agent.graph.detect_companies", lambda query: ["srf"])
    planner = json.dumps({
        "steps": [{"tool": "search_index", "args": {"query": "SRF revenue", "k": 5}}],
    })
    answer = json.dumps({
        "answer": "SRF revenue was 100.",
        "citations": ["c1"],
        "insufficient_evidence": False,
    })

    def fake_search(query: str, k: int = 5, **kwargs) -> dict:
        return {
            "hits": [{
                "chunk_id": "c1",
                "meta": {"company_id": "deepak_nitrite"},
                "text": "Revenue was 100.",
            }],
            "match_info": {},
        }

    graph = AgentGraph(
        llm=StubLLM([planner, _accept(), answer]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("SRF revenue", thread_id="t-company-mismatch")

    assert result["final_answer"]["insufficient_evidence"] is True
    assert result["final_answer"]["company_mismatches"] == ["srf"]
    assert any("company_mismatch: srf" in error for error in result["errors"])


def test_zero_hit_company_must_be_stated(monkeypatch):
    monkeypatch.setattr("src.agent.graph.detect_companies", lambda query: ["srf"])
    planner = json.dumps({
        "steps": [{"tool": "search_index", "args": {"query": "SRF revenue", "k": 5}}],
    })
    answer = json.dumps({
        "answer": "SRF revenue was 100.",
        "citations": [],
        "insufficient_evidence": False,
    })

    def fake_search(query: str, k: int = 5, **kwargs) -> dict:
        return {"hits": [], "match_info": {}}

    graph = AgentGraph(
        llm=StubLLM([planner, _accept(), answer]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("SRF revenue", thread_id="t-zero-hit")

    assert result["final_answer"]["insufficient_evidence"] is True
    assert result["final_answer"]["zero_hit_companies_not_stated"] == ["srf"]
    assert any("zero_hit_company_not_stated: srf" in e for e in result["errors"])


def test_synthesis_groups_chunks_by_company(monkeypatch):
    monkeypatch.setattr(
        "src.agent.graph.detect_companies", lambda query: ["srf", "deepak_nitrite"]
    )
    planner = json.dumps({
        "steps": [
            {"tool": "search_index", "args": {"query": "SRF revenue", "k": 5,
                                              "company": "srf"}},
            {"tool": "search_index", "args": {"query": "Deepak revenue", "k": 5,
                                              "company": "deepak_nitrite"}},
        ],
    })
    answer = json.dumps({
        "answer": "SRF: 100 [s1]. Deepak Nitrite: 200 [d1].",
        "citations": ["s1", "d1"],
        "insufficient_evidence": False,
    })

    def fake_search(query: str, k: int = 5, company: str | None = None, **kwargs) -> dict:
        return {
            "hits": [{
                "chunk_id": "s1" if company == "srf" else "d1",
                "meta": {"company_id": company},
                "text": "Revenue figure for the company.",
            }],
            "match_info": {},
        }

    llm = StubLLM([planner, _accept(), answer])
    graph = AgentGraph(
        llm=llm,
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("SRF vs Deepak Nitrite revenue", thread_id="t-groups")

    answer_prompt = next(prompt for prompt in llm.calls if "Chunks:\n" in prompt)
    assert "COMPANY: srf" in answer_prompt
    assert "COMPANY: deepak_nitrite" in answer_prompt
    assert "Companies with zero hits: []" in answer_prompt
    assert result["final_answer"]["insufficient_evidence"] is False

def test_answer_chunks_keep_tail_and_cap():
    from src.agent import graph as graph_mod
    long_text = "HEAD " + ("x" * 1600) + "\nTAIL_ROW Total assets 100"
    chunks = graph_mod._answer_chunks([{
        "tool": "search_index",
        "result": {"hits": [{"chunk_id": "c1", "text": long_text, "meta": {"company_id": "acme"}}]},
    }], query="total assets", max_chunks=8)
    assert len(chunks) == 1
    assert "TAIL_ROW Total assets 100" in chunks[0]["text"]
    assert "HEAD" not in chunks[0]["text"]


def test_basis_filter_excludes_unclear(monkeypatch, tmp_path):
    from src.agent import tools as agent_tools
    index = _FakeIndex([
        {"chunk_id": "c1", "text": "Total assets 100", "meta": {"company_id": "acme", "period": "FY2025", "statement_kind": "balance_sheet", "reporting_basis": "unclear"}},
        {"chunk_id": "c2", "text": "Total assets 200", "meta": {"company_id": "acme", "period": "FY2025", "statement_kind": "balance_sheet", "reporting_basis": "consolidated"}},
    ])
    _fake_index(monkeypatch, index)
    out = agent_tools.search_index("total assets", k=5, company="acme", period="FY2025", statement_kind="balance_sheet", reporting_basis="consolidated")
    ids = [h["chunk_id"] for h in out["hits"]]
    assert ids == ["c2"]


def test_wrong_company_figure_rejected(monkeypatch):
    index = _FakeIndex([
        {"chunk_id": "c1", "text": "Total assets 100", "meta": {"company_id": "acme", "period": "FY2025", "statement_kind": "balance_sheet", "reporting_basis": "consolidated"}},
    ])
    _fake_index(monkeypatch, index)
    llm = StubLLM([
        json.dumps({"steps":[{"tool":"search_index","args":{"query":"total assets","k":5,"company":"acme","period":"FY2025","statement_kind":"balance_sheet","reporting_basis":"consolidated"}}]}),
        _accept(),
        json.dumps({
            "answer": "acme total assets 100",
            "citations": ["c1"],
            "insufficient_evidence": False,
            "figures": [{"company_id":"other","label":"Total assets","period_column":"current","value":"100","chunk_id":"c1"}]
        }),
    ])
    g = AgentGraph(llm=llm, checkpointer=MemorySaver())
    res = g.run("What is total assets for acme FY2025 consolidated?", thread_id="twcf")
    assert res["final_answer"]["insufficient_evidence"] is True
    assert any("ungrounded_figure: other/Total assets" in e for e in res["errors"])


def test_difference_computed_in_code(monkeypatch):
    index = _FakeIndex([
        {"chunk_id": "a1", "text": "Total assets 200", "meta": {"company_id": "alpha", "period": "FY2025", "statement_kind": "balance_sheet", "reporting_basis": "consolidated"}},
        {"chunk_id": "b1", "text": "Total assets 150", "meta": {"company_id": "beta", "period": "FY2025", "statement_kind": "balance_sheet", "reporting_basis": "consolidated"}},
    ])
    _fake_index(monkeypatch, index)
    llm = StubLLM([
        json.dumps({"steps":[
            {"tool":"search_index","args":{"query":"total assets alpha","k":5,"company":"alpha","period":"FY2025","statement_kind":"balance_sheet","reporting_basis":"consolidated"}},
            {"tool":"search_index","args":{"query":"total assets beta","k":5,"company":"beta","period":"FY2025","statement_kind":"balance_sheet","reporting_basis":"consolidated"}},
        ]}),
        _accept(),
        json.dumps({
            "answer": "ignored",
            "citations": ["a1","b1"],
            "insufficient_evidence": False,
            "figures": [
                {"company_id":"alpha","label":"Total assets","period_column":"current","value":"200","chunk_id":"a1"},
                {"company_id":"beta","label":"Total assets","period_column":"current","value":"150","chunk_id":"b1"},
            ]
        }),
    ])
    g = AgentGraph(llm=llm, checkpointer=MemorySaver())
    res = g.run("Compare alpha and beta on total assets, by how much?", thread_id="tdiff")
    assert "difference 50" in res["final_answer"]["answer"]
    assert "alpha" in res["final_answer"]["answer"].lower()


def test_alias_fix_no_l_and_t_false_positive(monkeypatch):
    from src.agent import tools as agent_tools
    monkeypatch.setattr(agent_tools, "company_ids", lambda path=None: ("larsen_toubro", "tcs"))
    got = agent_tools.detect_companies("compare l and t revenue")
    assert "larsen_toubro" not in got


def test_company_ids_does_not_read_chunks_jsonl(tmp_path, monkeypatch):
    (tmp_path / "company_ids.json").write_text(
        json.dumps({"company_ids": ["acme", "beta"]}), encoding="utf-8"
    )
    real_open = Path.open

    def guarded_open(self, *args, **kwargs):
        if "chunks.jsonl" in str(self):
            raise AssertionError("company_ids must not read chunks.jsonl")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    got = tools.company_ids(str(tmp_path))
    assert got == ("acme", "beta")


def test_t2g_two_correct_citations_not_flagged(monkeypatch):
    index = _FakeIndex([
        {"chunk_id": "a1", "text": "Total equity 326469.3", "meta": {"company_id": "aurobindo_pharma", "period": "FY2025", "statement_kind": "balance_sheet", "reporting_basis": "consolidated"}},
        {"chunk_id": "d1", "text": "Total equity 288566", "meta": {"company_id": "dr_reddys", "period": "FY2025", "statement_kind": "balance_sheet", "reporting_basis": "consolidated"}},
    ])
    _fake_index(monkeypatch, index)
    monkeypatch.setattr("src.agent.graph.detect_companies", lambda query, unresolved=None, logged_unresolved=None: ["aurobindo_pharma", "dr_reddys"])
    llm = StubLLM([
        json.dumps({"steps":[
            {"tool":"search_index","args":{"query":"total equity aurobindo","k":5,"company":"aurobindo_pharma","period":"FY2025","statement_kind":"balance_sheet","reporting_basis":"consolidated"}},
            {"tool":"search_index","args":{"query":"total equity dr reddys","k":5,"company":"dr_reddys","period":"FY2025","statement_kind":"balance_sheet","reporting_basis":"consolidated"}},
        ]}),
        _accept(),
        json.dumps({
            "answer": "aurobindo_pharma total equity 326469.3 and dr_reddys total equity 288566",
            "citations": ["a1","d1"],
            "insufficient_evidence": False,
            "figures": [
                {"company_id":"aurobindo_pharma","label":"Total equity","period_column":"current","value":"326469.3","chunk_id":"a1"},
                {"company_id":"dr_reddys","label":"Total equity","period_column":"current","value":"288566","chunk_id":"d1"},
            ]
        }),
    ])
    g = AgentGraph(llm=llm, checkpointer=MemorySaver())
    res = g.run("For the year ended 31 March 2025, consolidated (both companies): is total equity larger for aurobindo or for Dr reddys, and what are the two figures in Rs. in million as printed?", thread_id="t2gtest")
    assert res["final_answer"].get("company_mismatches", []) == []
    assert res["final_answer"].get("insufficient_evidence") is False

def test_total_assets_accepts_total_equity_and_liabilities():
    from src.agent import graph as graph_mod
    tool_results = [{
        "tool": "search_index",
        "result": {"hits": [{
            "chunk_id": "c1",
            "text": "Total equity and liabilities 100",
            "meta": {"company_id": "acme", "reporting_basis": "consolidated"},
        }]},
    }]
    answer = {
        "answer": "acme total assets 100",
        "citations": ["c1"],
        "insufficient_evidence": False,
        "figures": [{
            "company_id": "acme",
            "label": "Total assets",
            "period_column": "current",
            "value": "100",
            "chunk_id": "c1",
        }],
    }
    errors: list[str] = []
    out = graph_mod._verify_structured_figures(
        answer, tool_results, errors, "acme consolidated total assets"
    )
    assert errors == []
    assert out["figures"][0]["label_equivalence"] is True


def test_wrong_company_citation_is_rejected(monkeypatch):
    monkeypatch.setattr("src.agent.graph.detect_companies", lambda query: ["srf"])
    planner = json.dumps({
        "steps": [{"tool": "search_index", "args": {"query": "SRF revenue", "k": 5}}],
    })
    answer = json.dumps({
        "answer": "SRF revenue was 100 [c1 p.1].",
        "citations": ["c1"],
        "insufficient_evidence": False,
        "figures": [
            {"company_id": "srf", "label": "Revenue", "period_column": "current",
             "value": "100", "chunk_id": "c1"},
        ],
    })

    def fake_search(query: str, k: int = 5, **kwargs) -> dict:
        return {
            "hits": [{
                "chunk_id": "c1",
                "meta": {"company_id": "deepak_nitrite"},
                "text": "Revenue was 100.",
            }],
            "match_info": {},
        }

    graph = AgentGraph(
        llm=StubLLM([planner, _accept(), answer]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("SRF revenue", thread_id="t-wrong-company-citation")

    assert result["final_answer"]["insufficient_evidence"] is True
    assert "srf" in result["final_answer"]["company_mismatches"]
    assert any("company_mismatch: srf" in error for error in result["errors"])


def test_plan_enforcement_adds_calculator_for_growth_queries():
    assert _growth_requested("What is Coforge growth in %?")
    steps, notes = _enforce_plan_tools(
        [{"tool": "search_index", "args": {"query": "Revenue from operations"}}],
        "What is Coforge growth in %?",
        [],
        remaining=10,
    )
    assert any(step.get("tool") == "calculator" for step in steps)
    assert any("enforced calculator" in note for note in notes)


def test_plan_enforcement_adds_get_chunk_when_top_hit_lacks_unit():
    tool_results = [{
        "tool": "search_index",
        "result": {"hits": [{"chunk_id": "abc:t1", "text": "Revenue from operations 100"}]},
    }]
    steps, notes = _enforce_plan_tools(
        [{"tool": "search_index", "args": {"query": "Revenue from operations"}}],
        "What is revenue?",
        tool_results,
        remaining=10,
    )
    get_steps = [step for step in steps if step.get("tool") == "get_chunk"]
    assert get_steps and get_steps[0]["args"]["chunk_id"] == "abc:t1"
    assert any("enforced get_chunk" in note for note in notes)


def test_alias_resolution_ignores_reading_prose() -> None:
    ids = ["adani_green", "srf"]
    unresolved: list[str] = []
    assert tools.detect_companies(
        "What is the Total assets reading for Adani Green?", ids=ids, unresolved=unresolved
    ) == ["adani_green"]
    assert "reading" not in unresolved
