"""Unit tests for page-level retriever (LLD-29 Day 5)."""
from __future__ import annotations

import json

import numpy as np

from src.tools import page_retrieve as pr


def _fake_records():
    return [
        {"doc_id": "a" * 64, "page": 1, "window": 0, "text": "revenue 100", "id": "a::p1::w0"},
        {"doc_id": "a" * 64, "page": 2, "window": 0, "text": "assets 200", "id": "a::p2::w0"},
        {"doc_id": "b" * 64, "page": 1, "window": 0, "text": "revenue 300", "id": "b::p1::w0"},
    ]


def test_windows_split_and_map():
    text = "x" * 3000
    ws = pr._windows(text, max_chars=1400)
    assert len(ws) == 2
    assert all(len(w) <= 1400 for w in ws)


def test_windows_short():
    assert pr._windows("short", max_chars=1400) == ["short"]


def test_build_page_records_maps_windows(tmp_path, monkeypatch):
    def fake_read(storage_dir, doc_id):
        return [
        {"pdf_page_index": 1, "page_text": "hello"},
        {"pdf_page_index": 2, "page_text": "y" * 3000},
    ]
    monkeypatch.setattr(pr, "read_page_extracts", fake_read)
    recs = pr.build_page_records(["doc1"])
    assert {(r["page"], r["window"]) for r in recs} == {(1, 0), (2, 0), (2, 1)}
    assert all(r["doc_id"] == "doc1" for r in recs)


def test_rrf_merge():
    out = pr._rrf([[(1, 10.0), (2, 5.0)], [(2, 1.0), (1, 0.5)]], k=60, topn=2)
    assert [i for i, _ in out] == [1, 2]


def test_page_retriever_cache_hit(tmp_path, monkeypatch):
    recs = _fake_records()
    (tmp_path / "page_records.json").write_text(json.dumps(recs))
    emb = np.eye(3, dtype=np.float32)
    np.save(tmp_path / "embeddings.npy", emb)
    cfg = {"hybrid": {"rrf_k": 60},
           "rerank": {"top_k": 5, "model": "Xenova/ms-marco-MiniLM-L-6-v2"},
           "page": {"rerank_pool": 20}}
    r = pr.PageRetriever(config=cfg, cache=tmp_path)
    assert len(r.records) == 3
    assert r.embeddings.shape == (3, 3)


def test_multi_company_merge(monkeypatch, tmp_path):
    recs = _fake_records()
    (tmp_path / "page_records.json").write_text(json.dumps(recs))
    np.save(tmp_path / "embeddings.npy", np.random.rand(3, 384).astype(np.float32))
    cfg = {"hybrid": {"rrf_k": 60},
           "rerank": {"top_k": 5, "model": "Xenova/ms-marco-MiniLM-L-6-v2"},
           "page": {"rerank_pool": 20}}
    r = pr.PageRetriever(config=cfg, cache=tmp_path)
    monkeypatch.setattr(r, "_search_once", lambda q, allowed_docs=None: [
        pr.RetrievedChunk("a"*64,1,"t",1.0), pr.RetrievedChunk("a"*64,2,"t2",0.9),
        pr.RetrievedChunk("b"*64,1,"t3",0.8), pr.RetrievedChunk("b"*64,2,"t4",0.7)])
    monkeypatch.setattr(pr, "resolve_query", lambda q: type("R", (), {"company_ids":["c1","c2"]})())
    reg = type("Reg", (), {"by_id": {"c1": {"display_name": "A"}, "c2": {"display_name": "B"}}})()
    monkeypatch.setattr("src.resolver.query_resolver.load_company_registry", lambda p: reg)
    out = r._retrieve_multi_company("compare A and B revenue", ["c1","c2"])
    assert len(out) <= 5
    assert len({(c.doc_id,c.page) for c in out}) == len(out)
