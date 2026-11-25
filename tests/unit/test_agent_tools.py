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


def test_query_embedding_retries_timeout_and_caches(monkeypatch):
    calls = []

    def fake_embed(text, dim, embedder):
        calls.append((text, dim, embedder))
        if len(calls) == 1:
            raise RequestsTimeout("embedding timed out")
        return np.full(dim, 2.0, dtype=np.float32)

    monkeypatch.setattr(fi, "embed_text", fake_embed)
    index = _FakeIndex("bge-small")
    first = tools._query_embedding("revenue", index, "dummy")
    second = tools._query_embedding("revenue", index, "dummy")
    assert len(calls) == 2
    assert np.array_equal(first, second)
    assert first[0] == 2.0
