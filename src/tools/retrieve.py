"""Track B retrieval tool: cached C2 index + cleaned BM25 hybrid, RRF, MiniLM rerank.

Reads configs/retrieval.yaml and results/week2_retrieval_cache/C2. The corpus
embeddings are loaded from disk; no corpus chunk is ever re-embedded.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml

from src.rag_ingestion.contract import read_page_extracts
from src.rag_ingestion.index import cosine_search
from src.rag_ingestion.tokenizer import tokenize as base_tokenize

CONFIG_PATH = Path("configs/retrieval.yaml")
CACHE = Path("results/week2_retrieval_cache/C2")
STOPWORDS = set(
    """a an and are as at be by for from has have he in is it its of on or that the to was were
    will with this these those but if into no not than then so such can may we our their they you
    your i me my""".split()
)
_CHUNK_RE = re.compile(r"^(?P<doc>.+)::C2::p(?P<page>\d+)::w(?P<win>\d+)$")


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    page: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.doc_id[:12]} p.{self.page}]"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def cleaned_tokens(text: str) -> list[str]:
    tokens = base_tokenize(text or "")
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def embed_query(text: str, *, config: dict[str, Any] | None = None) -> np.ndarray:
    """Embed one NEW query on VM02; never re-embeds a cached corpus chunk."""
    config = config or load_config()
    base = os.environ.get("LLM_EMBEDDINGS_URL")
    if not base:
        raise RuntimeError("LLM_EMBEDDINGS_URL is not configured")
    model = os.environ.get("EMBEDDINGS_MODEL", config["dense"]["model"])
    key = os.environ.get("EMBEDDINGS_API_KEY") or os.environ.get("LLM_API_KEY")
    ca = os.environ.get("EMBEDDINGS_CA_CERT")
    verify = ca if ca and Path(ca).exists() else True
    response = requests.post(
        base.rstrip("/") + "/embeddings",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "input": [text]},
        timeout=60,
        verify=verify,
    )
    response.raise_for_status()
    data = response.json()["data"]
    return np.asarray(data[0]["embedding"], dtype=np.float32)


def _rrf(rankings: list[list[tuple[int, float]]], k: int, topn: int) -> list[tuple[int, float]]:
    agg: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking, start=1):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda item: (-item[1], item[0]))[:topn]


class Retriever:
    """Hybrid retriever backed exclusively by the frozen C2 caches."""

    def __init__(self, config: dict[str, Any] | None = None, cache: Path = CACHE) -> None:
        self.config = config or load_config()
        self.cache = Path(cache)
        self.ids = json.loads((self.cache / "chunk_ids.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(self.cache / self.config["dense"]["embeddings"].split("/")[-1])
        if self.embeddings.shape[0] != len(self.ids):
            raise ValueError("cached embeddings and chunk ids disagree")
        self.k = int(self.config["hybrid"]["rrf_k"])
        self.pool = int(self.config["hybrid"]["candidate_pool_k"])
        self.top_k = int(self.config["rerank"]["top_k"])
        self._pages: dict[tuple[str, int], dict[int, str]] = {}
        self._model = None
        self._lock = threading.Lock()

    def _bm25(self) -> Any:
        from rank_bm25 import BM25Okapi
        if not hasattr(self, "_bm25_instance"):
            self._bm25_instance = BM25Okapi([cleaned_tokens(self.chunk_text(i))
                                             for i in range(len(self.ids))])
        return self._bm25_instance

    def chunk_text(self, index: int) -> str:
        match = _CHUNK_RE.match(self.ids[index])
        if not match:
            raise ValueError(f"unexpected chunk id: {self.ids[index]}")
        doc, page, win = match.group("doc"), int(match.group("page")), int(match.group("win"))
        pages = self._pages.setdefault(doc, {})
        if page not in pages:
            records = read_page_extracts("data/corpus", doc)
            step = 800 - 100
            for record in records:
                p = record.get("pdf_page_index")
                text = record.get("page_text") or ""
                if not isinstance(p, int) or not text.strip():
                    continue
                windows = {}
                for j, start in enumerate(range(0, len(text), step)):
                    piece = text[start:start + 800]
                    if piece.strip():
                        windows[j] = piece
                pages[p] = windows
        return pages.get(page, {}).get(win, "")

    def _rerank_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                    self._model = TextCrossEncoder(
                        model_name=self.config["rerank"]["model"], threads=4,
                    )
        return self._model

    def _rerank(self, query: str, pool: list[tuple[int, float]]) -> list[tuple[int, float]]:
        try:
            scores = list(self._rerank_model().rerank(query, [self.chunk_text(i)
                                                              for i, _ in pool]))
        except Exception:
            return pool[:self.top_k]
        ordered = sorted(zip([i for i, _ in pool], scores), key=lambda item: -float(item[1]))
        return ordered[:self.top_k]

    def retrieve(
        self, query: str, *, query_vector: np.ndarray | None = None,
    ) -> list[RetrievedChunk]:
        bm25 = self._bm25()
        scores = bm25.get_scores(cleaned_tokens(query))
        top = min(self.pool, len(scores))
        idx = np.argpartition(-scores, top - 1)[:top]
        idx = idx[np.argsort(-scores[idx])]
        bm25_rank = [(int(i), float(scores[i])) for i in idx]
        vector = (query_vector if query_vector is not None
                  else embed_query(query, config=self.config))
        dense_rank = cosine_search(self.embeddings, vector, k=self.pool)
        pool = _rrf([bm25_rank, dense_rank], self.k, self.pool)
        reranked = self._rerank(query, pool)
        out = []
        for index, score in reranked:
            match = _CHUNK_RE.match(self.ids[index])
            if not match:
                continue
            out.append(RetrievedChunk(
                doc_id=match.group("doc"), page=int(match.group("page")),
                text=self.chunk_text(index), score=float(score),
            ))
        return out


_default: Retriever | None = None
_page_default = None


def get_retriever(config: dict[str, Any] | None = None):
    cfg = config or load_config()
    mode = str(cfg.get("retriever", "chunk"))
    if mode == "page":
        global _page_default
        if _page_default is None:
            from src.tools.page_retrieve import PageRetriever
            _page_default = PageRetriever(cfg)
        return _page_default, mode
    global _default
    if _default is None:
        _default = Retriever(cfg)
    return _default, "chunk"


def retrieve(query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
    retriever, mode = get_retriever()
    chunks = retriever.retrieve(query)
    if top_k is not None:
        return chunks[:top_k]
    return chunks
