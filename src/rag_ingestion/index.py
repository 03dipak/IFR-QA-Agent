"""Index builder: BM25 (own implementation, no dependency) + dense embeddings
(fastembed, CPU). Reads chunks from src.rag_ingestion.chunking only.

BM25 dependency note: own implementation from Robertson/Zaragoza (k1=1.5,
b=0.75, idf = ln((N-df+0.5)/(df+0.5)+1), token = lowercase [a-z0-9]+).
Dense dependency: fastembed (CPU-only ONNX); model stated in index config.
"""

from __future__ import annotations

import json
import math
import re
from .tokenizer import tokenize as _shared_tokenize, TOKENIZER_VERSION
import time
from array import array
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    return _shared_tokenize(text)


class BM25Index:
    """Memory-compact BM25: integer token ids, array postings, metadata-only chunks."""

    def __init__(self, chunks: list[dict]):
        t0 = time.time()
        self.chunks = [{"meta": c.get("meta", {})} for c in chunks]
        self.N = len(chunks)
        self.vocab: dict[str, int] = {}
        self.dl = array("i", [0]) * self.N if self.N else array("i")
        df: Counter = Counter()
        for i, c in enumerate(chunks):
            ids = [self.vocab.setdefault(t, len(self.vocab)) for t in tokenize(c["text"])]
            self.dl[i] = len(ids)
            for t in set(ids):
                df[t] += 1
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.idf = {t: math.log((self.N - d + 0.5) / (d + 0.5) + 1.0)
                    for t, d in df.items()}
        post_ids: dict[int, array] = defaultdict(lambda: array("i"))
        post_tfs: dict[int, array] = defaultdict(lambda: array("i"))
        for i, c in enumerate(chunks):
            ids = [self.vocab[t] for t in tokenize(c["text"])]
            tf = Counter(ids)
            for t, f in tf.items():
                post_ids[t].append(i)
                post_tfs[t].append(f)
        self.postings = {t: (post_ids[t], post_tfs[t]) for t in post_ids}
        self.dl_arr = np.asarray(self.dl, dtype=np.float32)
        self.build_seconds = time.time() - t0

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = np.zeros(self.N, dtype=np.float32)
        for t in tokenize(query):
            tid = self.vocab.get(t)
            if tid is None:
                continue
            entry = self.postings.get(tid)
            if not entry:
                continue
            ids, tfs = entry
            idx = np.asarray(ids, dtype=np.int64)
            f = np.asarray(tfs, dtype=np.float32)
            num = f * (K1 + 1)
            den = f + K1 * (1 - B + B * self.dl_arr[idx] / (self.avgdl or 1))
            scores[idx] += self.idf[tid] * num / den
        kk = min(k, self.N)
        if kk <= 0:
            return []
        top = np.argpartition(-scores, kk - 1)[:kk]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


def build_bm25(chunks: list[dict]) -> BM25Index:
    return BM25Index(chunks)


def build_dense(chunks: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Returns (embeddings ndarray[L2-normalised], model_id, build_seconds)."""
    from fastembed import TextEmbedding
    t0 = time.time()
    model = TextEmbedding(model_name=model_name)
    texts = [c["text"] for c in chunks]
    emb = list(model.embed(texts, batch_size=256))
    import numpy as np
    arr = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norm == 0, 1.0, norm)
    return arr, model_name, time.time() - t0


def cosine_search(arr, query_vec, k: int = 10) -> list[tuple[int, float]]:
    import numpy as np
    q = query_vec.astype(np.float32)
    n = np.linalg.norm(q)
    q = q / (n if n else 1.0)
    scores = arr @ q
    idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


class DenseIndex:
    def __init__(self, chunks: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.arr, self.model_id, self.build_seconds = build_dense(chunks, model_name)
        self._model = None

    def _embed(self, query: str):
        from fastembed import TextEmbedding
        import numpy as np
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_id)
        vec = next(self._model.embed([query]))
        return np.asarray(vec, dtype=np.float32)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        return cosine_search(self.arr, self._embed(query), k)


def save_index(chunks: list[dict], variant: str, index_version: str,
               root: str = "data/index_v1") -> Path:
    out = Path(root) / f"{variant}_{index_version}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, sort_keys=True) + "\n")
    return out
