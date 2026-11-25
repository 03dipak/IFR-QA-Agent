"""Page-level retriever for dev docs (LLD-29 Day 5). Read-only over cached index."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml

from src.rag_ingestion.contract import read_page_extracts
from src.rag_ingestion.tokenizer import tokenize as base_tokenize
from src.resolver.query_resolver import resolve_query

CONFIG_PATH = Path("configs/retrieval.yaml")
PAGE_CACHE = Path("results/page_index_cache")
STOPWORDS = set("""a an and are as at be by for from has have he in is it its of on or that the to
was were will with this these those but if into no not than then so such can may we our their they
you your i me my""".split())


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    page: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.doc_id[:12]} p.{self.page}]"


def cleaned_tokens(text: str) -> list[str]:
    return [t for t in base_tokenize(text or "") if len(t) > 1 and t not in STOPWORDS]


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _embed_texts(texts: list[str], config: dict[str, Any]) -> np.ndarray:
    base = os.environ.get("LLM_EMBEDDINGS_URL")
    if not base:
        raise RuntimeError("LLM_EMBEDDINGS_URL is not configured")
    model = os.environ.get(
        "EMBEDDINGS_MODEL", config.get("dense", {}).get("model", "bge-small-en-v1.5")
    )
    key = os.environ.get("EMBEDDINGS_API_KEY") or os.environ.get("LLM_API_KEY")
    ca = os.environ.get("EMBEDDINGS_CA_CERT")
    verify = ca if ca and Path(ca).exists() else True
    resp = requests.post(
        base.rstrip("/") + "/embeddings",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "input": texts},
        timeout=120, verify=verify,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    data = sorted(data, key=lambda x: x["index"])
    return np.asarray([d["embedding"] for d in data], dtype=np.float32)


def _windows(text: str, max_chars: int = 1400) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    # split into 2 windows, both mapped to same page
    mid = len(text) // 2
    w1 = text[:mid].strip()
    w2 = text[mid:].strip()
    out = []
    for w in [w1, w2]:
        if len(w) <= max_chars:
            out.append(w)
        else:
            out.append(w[:max_chars].strip())
    return [w for w in out if w]


def build_page_records(
    doc_ids: list[str], storage_dir: str = "data/corpus"
) -> list[dict[str, Any]]:
    records = []
    for doc_id in doc_ids:
        for rec in read_page_extracts(storage_dir, doc_id):
            page = rec.get("pdf_page_index")
            text = rec.get("page_text") or ""
            if not isinstance(page, int) or not text.strip():
                continue
            for wi, win in enumerate(_windows(text)):
                records.append({"doc_id": doc_id, "page": page, "window": wi, "text": win,
                                "id": f"{doc_id}::p{page}::w{wi}"})
    records.sort(key=lambda r: (r["doc_id"], r["page"], r["window"]))
    return records


def build_page_index(
    doc_ids: list[str], cache: Path = PAGE_CACHE, config: dict[str, Any] | None = None,
                     batch_size: int = 16) -> dict[str, Any]:
    config = config or load_config()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    rec_path = cache / "page_records.json"
    emb_path = cache / "embeddings.npy"
    records = build_page_records(doc_ids)
    rec_path.write_text(json.dumps(records), encoding="utf-8")
    # resume
    if emb_path.exists():
        arr = np.load(emb_path)
        if arr.shape[0] == len(records):
            n_pages = len({(r["doc_id"], r["page"]) for r in records})
            return {"n_records": len(records), "n_pages": n_pages, "cached": True}
    done = 0
    vecs = []
    if emb_path.exists():
        old = np.load(emb_path)
        done = old.shape[0]
        vecs = [old]
    texts = [r["text"] for r in records]
    for i in range(done, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            emb = _embed_texts(batch, config)
        except Exception:
            rows = []
            for t in batch:
                try:
                    rows.append(_embed_texts([t], config))
                except Exception:
                    rows.append(np.zeros((1, 384), dtype=np.float32))
            emb = np.vstack(rows).astype(np.float32)
        vecs.append(emb)
        arr = np.vstack(vecs).astype(np.float32)
        np.save(emb_path, arr)
        if (i // batch_size) % 10 == 0:
            print(f"embedded {min(i+batch_size,len(texts))}/{len(texts)}", flush=True)
    arr = np.load(emb_path)
    n_pages = len({(r["doc_id"], r["page"]) for r in records})
    return {
        "n_records": len(records), "n_pages": n_pages,
        "cached": False, "shape": list(arr.shape),
    }


def _rrf(rankings: list[list[tuple[int, float]]], k: int, topn: int) -> list[tuple[int, float]]:
    agg: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking, start=1):
            agg[idx] = agg.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda item: (-item[1], item[0]))[:topn]


class PageRetriever:
    def __init__(self, config: dict[str, Any] | None = None, cache: Path = PAGE_CACHE):
        self.config = config or load_config()
        self.cache = Path(cache)
        self.records = json.loads((self.cache / "page_records.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(self.cache / "embeddings.npy")
        if self.embeddings.shape[0] != len(self.records):
            raise ValueError("page embeddings and records disagree")
        self.k = int(self.config.get("hybrid", {}).get("rrf_k", 60))
        self.top_k = int(self.config.get("rerank", {}).get("top_k", 5))
        self.page_rerank_pool = int(
            self.config.get("page", {}).get("rerank_pool", 20)
        )
        self._model = None
        self._lock = threading.Lock()
        self._bm25 = None

    def _bm25_model(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([cleaned_tokens(r["text"]) for r in self.records])
        return self._bm25

    def _rerank_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                    self._model = TextCrossEncoder(
                        model_name=self.config.get("rerank", {}).get(
                            "model", "Xenova/ms-marco-MiniLM-L-6-v2"
                        ),
                        threads=4,
                    )
        return self._model

    def _search_once(
        self, query: str, allowed_docs: set[str] | None = None
    ) -> list[RetrievedChunk]:
        bm25 = self._bm25_model()
        scores = bm25.get_scores(cleaned_tokens(query))
        if allowed_docs is not None:
            mask = np.array([r["doc_id"] in allowed_docs for r in self.records], dtype=bool)
            scores = np.where(mask, scores, -1e9)
        top = min(200, len(scores))
        idx = np.argpartition(-scores, top - 1)[:top]
        idx = idx[np.argsort(-scores[idx])]
        bm25_rank = [
            (int(i), float(scores[i])) for i in idx if scores[i] > -1e8
        ]
        q = _embed_texts([query], self.config)[0].astype(np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        dense_scores = (self.embeddings @ q).astype(np.float32)
        if allowed_docs is not None:
            dense_scores = np.where(mask, dense_scores, -1e9)
        dtop = min(200, len(dense_scores))
        didx = np.argpartition(-dense_scores, dtop - 1)[:dtop]
        didx = didx[np.argsort(-dense_scores[didx])]
        dense_rank = [
            (int(i), float(dense_scores[i])) for i in didx if dense_scores[i] > -1e8
        ]
        pool = _rrf([bm25_rank, dense_rank], self.k, 200)
        # top-3 docs by best page score
        doc_best: dict[str, float] = {}
        for i, s in pool:
            d = self.records[i]["doc_id"]
            doc_best[d] = max(doc_best.get(d, -1e9), s)
        keep_docs = {d for d, _ in sorted(doc_best.items(), key=lambda x: -x[1])[:3]}
        pool = [(i, s) for i, s in pool if self.records[i]["doc_id"] in keep_docs]
        # rerank top-20 pages
        pool = pool[: self.page_rerank_pool]
        try:
            texts = [self.records[i]["text"] for i, _ in pool]
            rerank_scores = list(self._rerank_model().rerank(query, texts))
        except Exception:
            rerank_scores = [s for _, s in pool]
        ordered = sorted(zip([i for i, _ in pool], rerank_scores), key=lambda x: -float(x[1]))
        # unique pages, keep best window
        seen = set()
        out = []
        for i, s in ordered:
            rec = self.records[i]
            key = (rec["doc_id"], rec["page"])
            if key in seen:
                continue
            seen.add(key)
            out.append(RetrievedChunk(rec["doc_id"], rec["page"], rec["text"], float(s)))
            if len(out) >= self.top_k:
                break
        return out

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        resolution = resolve_query(query)
        if len(resolution.company_ids) >= 2:
            return self._retrieve_multi_company(query, resolution.company_ids)
        return self._search_once(query)

    def _retrieve_multi_company(self, query: str, company_ids: list[str]) -> list[RetrievedChunk]:
        from src.resolver.query_resolver import load_company_registry
        reg = load_company_registry("data/acquisition/company_registry.csv")
        seen: set[tuple[str, int]] = set()
        out: list[RetrievedChunk] = []
        for cid in company_ids[:2]:
            row = reg.by_id.get(cid, {})
            cname = row.get("display_name") or cid
            chunks = self._search_once(f"{cname} {query}")
            for c in chunks[:3]:
                key = (c.doc_id, c.page)
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
        return out[: self.top_k]
