from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault("AGENT_INDEX_PATH", "results/full_index_v2/bge_small")
os.environ.setdefault("AGENT_RUN_TIMEOUT", "120")
os.environ.setdefault("AGENT_MAX_STEPS", "10")
os.environ.setdefault("AGENT_MAX_REVISIONS", "3")
os.environ.setdefault("AGENT_TOKEN_BUDGET", "12000")
os.environ.setdefault("LLM_CALL_TIMEOUT_SECONDS", "20")
os.environ.setdefault("EMBED_CALL_TIMEOUT_SECONDS", "10")
os.environ.setdefault("EXTERNAL_CALL_MAX_RETRIES", "2")
os.environ.setdefault("EXTERNAL_CALL_BACKOFF_SECONDS", "2,4")
os.environ.setdefault("LLM_CHAT_URL", "http://127.0.0.1:9")
os.environ.setdefault("LLM_MODEL", "stub")
os.environ.setdefault("LLM_API_KEY", "stub")

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src import request_deadline as budget
from src.agent import graph as agent_graph
from src.agent import service as service_mod
from src.agent import tools as tools_mod
from src.agent.llm import StubLLM


def test_get_metric_cache_reuses_result(monkeypatch):
    tools_mod.clear_get_metric_cache()
    calls = {"search": 0}

    def fake_search(*args, **kwargs):
        calls["search"] += 1
        return {
            "hits": [{"chunk_id": "d:p1:t0", "text": "Revenue from operations: 100", "meta": {"company_id": "demo", "period": "FY2025", "document_id": "d", "page": 1, "reporting_basis": "consolidated"}}],
            "match_info": {},
        }

    monkeypatch.setattr(tools_mod, "search_index", fake_search)
    import src.agent.extract as extract_mod
    monkeypatch.setattr(extract_mod, "extract_figures", lambda *a, **k: {"figures": [{"value": "100", "unit": "Rs. crore", "chunk_id": "d:p1:t0", "document_id": "d", "page": 1, "period": "FY2025", "reporting_basis": "consolidated", "label": "Revenue from operations"}], "missing": []})
    monkeypatch.setattr(extract_mod, "_unit", lambda chunk: "Rs. crore")
    first = tools_mod.get_metric("demo", "Revenue from operations", period="FY2025")
    second = tools_mod.get_metric("demo", "Revenue from operations", period="FY2025")
    assert first == second
    assert first.get("missing") is False
    assert calls["search"] == 1


def test_clarification_before_agent_loop(monkeypatch):
    tools_mod.clear_get_metric_cache()
    monkeypatch.setattr(tools_mod, "detect_companies", lambda q: [])
    monkeypatch.setattr(tools_mod, "detect_sector", lambda q: "Manufacturing")
    monkeypatch.setattr(tools_mod, "list_companies", lambda **kwargs: {"companies": ["a", "b", "c", "d"]})
    called = {"run": False}

    class BoomGraph:
        def __init__(self, *args, **kwargs):
            called["run"] = True

    monkeypatch.setattr(service_mod, "AgentGraph", BoomGraph)
    monkeypatch.setattr(service_mod, "EnvLLM", lambda: StubLLM([]))
    result = service_mod.run_question("Compare margins for manufacturing companies in FY2025")
    assert result.stop_reason == "clarification"
    assert "up to 3 companies" in result.answer.lower()
    assert called["run"] is False


def test_soft_deadline_returns_partial(monkeypatch):
    tools_mod.clear_get_metric_cache()

    def slow_tool(**kwargs):
        time.sleep(0.2)
        return {"value": 1}

    graph = agent_graph.AgentGraph(
        llm=StubLLM([]),
        tools={"slow_tool": slow_tool},
        checkpointer=MemorySaver(),
    )
    monkeypatch.setattr(budget, "soft_check", lambda margin_s=20.0: (_ for _ in ()).throw(TimeoutError("soft deadline reached (1.0s left)")))
    out = graph.run("test question")
    assert out["stop_reason"] in {"soft_deadline", "token_budget", "step_cap", "accepted", "tool_error"}


def test_three_company_comparison_partial(monkeypatch):
    tools_mod.clear_get_metric_cache()

    def fake_get_metric(company, metric, **kwargs):
        mapping = {
            "a": {"company_id": "a", "display_name": "A", "value": 10.0, "unit": "percent", "citation": "a:p1", "period": "FY2025", "missing": False, "derived": True, "reporting_basis": "consolidated"},
            "b": {"company_id": "b", "display_name": "B", "value": 20.0, "unit": "percent", "citation": "b:p1", "period": "FY2025", "missing": False, "derived": True, "reporting_basis": "consolidated"},
            "c": {"company_id": "c", "display_name": "C", "value": None, "unit": "percent", "citation": None, "period": "FY2025", "missing": True, "reason": "missing"},
        }
        return dict(mapping[company])

    monkeypatch.setattr(tools_mod, "list_companies", lambda **kwargs: {"companies": ["a", "b", "c"], "count": 3})
    monkeypatch.setattr(tools_mod, "get_metric", fake_get_metric)
    monkeypatch.setattr(tools_mod, "display_name", lambda cid: cid.upper())
    monkeypatch.setattr(tools_mod, "_sector_map", lambda: {"a": "Manufacturing", "b": "Manufacturing", "c": "Manufacturing"})
    result = tools_mod.rank_companies("operating profit margin", period="FY2025", sector="Manufacturing", n=3)
    assert result["count_found"] == 2
    assert len(result["missing"]) == 1
    assert result["ranked"][0]["company_id"] == "b"