import numpy as np
import pytest
from requests.exceptions import Timeout as RequestsTimeout

from src.agent import tools
from src.rag_ingestion import full_index as fi


@pytest.fixture(autouse=True)
def _clear_embedding_cache():
    tools._QUERY_EMBED_CACHE.clear()
    yield
    tools._QUERY_EMBED_CACHE.clear()


class _FakeIndex:
    def __init__(self, embedder: str):
        self.embedder = embedder
        self.meta = {"embedder": embedder, "dim": 4, "model": "local:hash384-v1" if embedder == "hash384" else "BAAI/bge-small-en-v1.5"}


def test_query_embedding_uses_hash384_without_prefix(monkeypatch):
    captured = {}

    def fake_embed(text, dim, embedder):
        captured["text"] = text
        captured["dim"] = dim
        captured["embedder"] = embedder
        return np.ones(dim, dtype=np.float32)

    monkeypatch.setattr(fi, "embed_text", fake_embed)
    index = _FakeIndex("hash384")
    vector = tools._query_embedding("revenue", index, "dummy")
    assert captured["text"] == "revenue"
    assert captured["embedder"] == "hash384"
    assert vector.shape == (4,)


def test_query_embedding_uses_bge_prefix(monkeypatch):
    captured = {}

    def fake_embed(text, dim, embedder):
        captured["text"] = text
        captured["dim"] = dim
        captured["embedder"] = embedder
        return np.ones(dim, dtype=np.float32)

    monkeypatch.setattr(fi, "embed_text", fake_embed)
    index = _FakeIndex("bge-small")
    vector = tools._query_embedding("revenue", index, "dummy")
    assert captured["text"] == fi.BGE_QUERY_PREFIX + "revenue"
    assert captured["embedder"] == "bge-small"
    assert vector.shape == (4,)


def test_query_embedding_propagates_timeout_and_caches(monkeypatch):
    calls = []

    def fake_embed(text, dim, embedder):
        calls.append((text, dim, embedder))
        if len(calls) == 1:
            raise RequestsTimeout("embedding timed out")
        return np.full(dim, 2.0, dtype=np.float32)

    monkeypatch.setattr(fi, "embed_text", fake_embed)
    index = _FakeIndex("bge-small")
    with pytest.raises(RequestsTimeout):
        tools._query_embedding("revenue", index, "dummy")
    first = tools._query_embedding("revenue", index, "dummy")
    second = tools._query_embedding("revenue", index, "dummy")
    assert len(calls) == 2
    assert np.array_equal(first, second)
    assert first[0] == 2.0


def test_search_index_reuses_exact_cached_result(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_SEARCH_CACHE_DIR", str(tmp_path / "search_cache"))
    monkeypatch.setattr(
        tools, "_query_embedding",
        lambda query, index, path: np.zeros(4, dtype=np.float32),
    )
    calls = []

    def fake_search_once(*args, **kwargs):
        calls.append(args)
        return (
            [{"chunk_id": "c1", "text": "revenue", "meta": {"company_id": "acme"}}],
            {"masked_rows": 1, "filter_mode": "masked"},
        )

    monkeypatch.setattr(tools, "_search_once", fake_search_once)
    first = tools.search_index("Revenue from operations", k=5, company="acme", period="FY2025")
    second = tools.search_index("Revenue from operations", k=5, company="acme", period="FY2025")

    assert len(calls) == 1
    assert second == first
    second["hits"][0]["text"] = "mutated"
    third = tools.search_index("Revenue from operations", k=5, company="acme", period="FY2025")
    assert third["hits"][0]["text"] == "revenue"


def test_search_cache_disabled_skips_read_and_write(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CACHE_ENABLED", "0")
    monkeypatch.setenv("AGENT_SEARCH_CACHE_DIR", str(tmp_path / "search_cache"))
    monkeypatch.setattr(tools, "_resolved_index_path", lambda: "dummy")
    monkeypatch.setattr(tools, "_open_index", lambda path=None: object())
    monkeypatch.setattr(
        tools, "_query_embedding",
        lambda query, index, path: np.zeros(4, dtype=np.float32),
    )
    calls = []

    def fake_search_once(*args, **kwargs):
        calls.append(args)
        return ([{"chunk_id": "c1", "text": "revenue", "meta": {}}],
                {"masked_rows": 1, "filter_mode": "masked"})

    monkeypatch.setattr(tools, "_search_once", fake_search_once)
    first = tools.search_index("Revenue from operations", k=5)
    second = tools.search_index("Revenue from operations", k=5)
    assert first == second
    assert len(calls) == 2
    assert not list((tmp_path / "search_cache").glob("**/*.json"))


def test_get_metric_rejects_lupin_note_segment_face_pl(monkeypatch):
    import importlib, src.agent.extract as ex, src.agent.tools as tools
    importlib.reload(ex); importlib.reload(tools)
    lupin_face = "58d072b738c8e25a064a5079ba368ca0b55049e88aa8ef74e1b891be70e035fe:p275"
    lupin_note = "70c9e12d1e37a08c43eb4e72836b4703c1506dc18d1fa501df6e8951770bd1b1:p354"
    lupin_seg = "58d072b738c8e25a064a5079ba368ca0b55049e88aa8ef74e1b891be70e035fe:p313"
    real_search = tools.search_index
    def fake_search(query, k=8, **kwargs):
        q=str(query).lower()
        if "lupin" in q and "revenue" in q:
            return [
                {"chunk_id": lupin_note, "score": 0.99, "text": "note", "meta": {"company_id":"lupin","period":kwargs.get("period") or "FY2024","reporting_basis":"consolidated","statement_kind":"income_statement","page":354}},
                {"chunk_id": lupin_seg, "score": 0.98, "text": "segment", "meta": {"company_id":"lupin","period":"FY2025","reporting_basis":"consolidated","statement_kind":"income_statement","page":313}},
                {"chunk_id": lupin_face, "score": 0.10, "text": "face", "meta": {"company_id":"lupin","period":"FY2025","reporting_basis":"consolidated","statement_kind":"income_statement","page":275}},
            ]
        return real_search(query, k=k, **kwargs)
    monkeypatch.setattr(tools, "search_index", fake_search)
    fy24 = tools.get_metric("lupin","Revenue from operations",period="FY2024")
    fy25 = tools.get_metric("lupin","Revenue from operations",period="FY2025")
    assert fy24["citation"] == lupin_face
    assert fy24["printed_value"] == "200,108.2"
    assert fy25["citation"] == lupin_face
    assert fy25["printed_value"] == "227,079.0"
