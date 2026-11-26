from __future__ import annotations

import os
import time
from types import SimpleNamespace

for _name, _value in {
    "BACKEND_TIMEOUT_SECONDS": "180",
    "AGENT_INDEX_PATH": "results/full_index_v2/bge_small",
    "AGENT_RUN_TIMEOUT": "120",
    "AGENT_MAX_STEPS": "10",
    "AGENT_MAX_REVISIONS": "3",
    "AGENT_TOKEN_BUDGET": "12000",
    "LLM_CALL_TIMEOUT_SECONDS": "20",
    "EMBED_CALL_TIMEOUT_SECONDS": "10",
    "EXTERNAL_CALL_MAX_RETRIES": "2",
    "EXTERNAL_CALL_BACKOFF_SECONDS": "2,4",
    "RERANK_MODEL": "Xenova/ms-marco-MiniLM-L-6-v2",
    "RERANK_CACHE_DIR": "models/fastembed",
    "RERANK_THREADS": "4",
}.items():
    os.environ.setdefault(_name, _value)

import pytest
import requests
from langgraph.checkpoint.memory import MemorySaver

from src import request_deadline as budget
from src.agent import graph as agent_graph
from src.agent import llm as llm_module
from src.agent import tools
from src.rag_ingestion import full_index
from src.ui.app import REQUEST_TIMEOUT


def test_ui_timeout_exceeds_agent_budget():
    assert REQUEST_TIMEOUT > 120.0


def test_deadline_caps_call_timeouts_and_expires():
    budget.start(0.2)
    try:
        assert budget.remaining(20.0) <= 20.0
        time.sleep(0.25)
        with pytest.raises(TimeoutError):
            budget.check()
        with pytest.raises(TimeoutError):
            budget.remaining(20.0)
    finally:
        budget.finish()


def test_llm_retries_are_bounded_by_two(monkeypatch):
    timeouts: list[float] = []

    def fake_post(url, **kwargs):
        timeouts.append(kwargs["timeout"])
        response = requests.Response()
        response.status_code = 503
        response._content = b"busy"
        response.url = url
        return response

    monkeypatch.setenv("LLM_CHAT_URL", "http://test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    monkeypatch.setattr(budget, "sleep", lambda seconds: None)
    budget.start(10.0)
    try:
        with pytest.raises(requests.HTTPError):
            llm_module.EnvLLM().complete("prompt")
    finally:
        budget.finish()
    assert len(timeouts) == 3
    assert all(0 < value <= 20.0 for value in timeouts)


def test_embedding_retries_are_bounded_by_two(monkeypatch, capsys):
    timeouts: list[float] = []

    def fake_post(url, **kwargs):
        timeouts.append(kwargs["timeout"])
        response = requests.Response()
        response.status_code = 503
        response._content = b"busy"
        response.url = url
        return response

    monkeypatch.setattr(full_index.requests, "post", fake_post)
    monkeypatch.setattr(budget, "sleep", lambda seconds: None)
    budget.start(10.0)
    try:
        with pytest.raises(requests.HTTPError):
            full_index._bge_request(
                ["x"], 384,
                {"LLM_EMBEDDINGS_URL": "http://test/v1", "LLM_API_KEY": "test-key"},
                True,
            )
    finally:
        budget.finish()
    capsys.readouterr()
    assert len(timeouts) == 3
    assert all(0 < value <= 10.0 for value in timeouts)


def test_reranker_failure_is_visible(monkeypatch):
    monkeypatch.setattr(tools, "_RERANK_MODEL", None)
    monkeypatch.setattr(tools, "_rerank_enabled", lambda: True)

    def broken_encoder(**kwargs):
        raise RuntimeError("missing onnx files")

    monkeypatch.setattr(
        "fastembed.rerank.cross_encoder.TextCrossEncoder", broken_encoder, raising=False
    )
    with pytest.raises(RuntimeError, match="reranker unavailable"):
        tools._rerank_model()


def test_agent_steps_stay_bounded():
    graph = agent_graph.AgentGraph(
        llm=SimpleNamespace(), checkpointer=MemorySaver()
    )
    assert graph.max_steps <= 10
