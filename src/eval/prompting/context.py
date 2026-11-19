"""B1 oracle context construction for Week-1 prompting."""

from __future__ import annotations

import json
from pathlib import Path
import hashlib
import math
from array import array
from typing import Iterable

from src.eval.contexts import _page_text
from src.eval.retrieval_sanity import _filter_chunks
from src.resolver.query_resolver import resolve_query, load_company_registry
from src.rag_ingestion.index import BM25Index, tokenize
from src.resolver.query_resolver import resolve_query, load_company_registry

PAGE_DIR = Path("data/corpus/extract/page_extracts")
CHUNKS: list[dict] | None = None
_CONTEXT_CACHE: dict[str, list[dict]] | None = None
_CONTEXT_CACHE_META: dict | None = None
_ABSTAIN_CACHE: dict[tuple, BM25Index] = {}
_ABSTAIN_CANDIDATE_CACHE: dict[tuple, list[dict]] = {}


def load_page_chunks() -> list[dict]:
    global CHUNKS
    if CHUNKS is not None:
        return CHUNKS
    chunks: list[dict] = []
    from src.eval.retrieval_sanity import freeze_docs
    for doc in freeze_docs():
        doc_id = doc["document_id"]
        path = PAGE_DIR / f"{doc_id}.pages.jsonl"
        if not path.is_file():
            continue
        usable = set()
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            records.append(rec)
            if rec.get("usable_as_evidence") and isinstance(rec.get("pdf_page_index"), int):
                usable.add(rec["pdf_page_index"])
        for rec in records:
            page = rec.get("pdf_page_index")
            if not isinstance(page, int) or page not in usable:
                continue
            text = rec.get("page_text") or ""
            if not text.strip():
                continue
            chunks.append({
                "id": f"{doc_id}:p{page}",
                "text": text,
                "meta": {
                    "document_id": doc_id,
                    "page": page,
                    "company_id": doc.get("company_id"),
                    "period": doc.get("fiscal_year"),
                    "variant": doc.get("variant"),
                    "doc_type": doc.get("doc_type"),
                    "chunk_type": "page",
                },
            })
    CHUNKS = chunks
    return chunks


class _LightBM25Index:
    """Memory-light BM25 restricted to a fixed query-term vocabulary."""
    K1, B = 1.5, 0.75

    def __init__(self, terms: set[str], docs: list[tuple[str, int]], dls: array,
                 postings: dict[str, tuple[array, array]], idfs: dict[str, float]):
        self.terms = terms
        self.docs = docs
        self.dls = dls
        self.postings = postings
        self.idfs = idfs
        self.N = len(docs)
        self.avgdl = (sum(dls) / self.N) if self.N else 0.0

    def search(self, query: str, k: int = 10):
        scores = [0.0] * self.N
        for t in tokenize(query):
            post = self.postings.get(t)
            if not post:
                continue
            ids, tfs = post
            for i, tf in zip(ids, tfs):
                dl = self.dls[i]
                den = tf + self.K1 * (1 - self.B + self.B * dl / (self.avgdl or 1))
                scores[i] += self.idfs[t] * tf * (self.K1 + 1) / den
        if not scores:
            return []
        top = sorted(range(self.N), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in top[:k]]


def _iter_page_chunks():
    from src.eval.retrieval_sanity import freeze_docs
    for doc in freeze_docs():
        doc_id = doc["document_id"]
        path = PAGE_DIR / f"{doc_id}.pages.jsonl"
        if not path.is_file():
            continue
        usable = set()
        records = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append(rec)
                if rec.get("usable_as_evidence") and isinstance(rec.get("pdf_page_index"), int):
                    usable.add(rec["pdf_page_index"])
        for rec in records:
            page = rec.get("pdf_page_index")
            if not isinstance(page, int) or page not in usable:
                continue
            text = rec.get("page_text") or ""
            if not text.strip():
                continue
            yield {"text": text,
                   "meta": {"document_id": doc_id, "page": page}}


def _light_global_index(rows: list[dict]) -> _LightBM25Index:
    terms: set[str] = set()
    for r in rows:
        terms.update(tokenize(r.get("question_text") or r.get("question") or ""))
    docs: list[tuple[str, int]] = []
    dls: array = array("I")
    postings: dict[str, tuple[array, array]] = {t: (array("I"), array("I")) for t in terms}
    for idx, ch in enumerate(_iter_page_chunks()):
        tokens = tokenize(ch["text"])
        docs.append((ch["meta"]["document_id"], ch["meta"]["page"]))
        dls.append(len(tokens))
        if tokens:
            counts: dict[str, int] = {}
            for t in tokens:
                if t in terms:
                    counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                ids, tfs = postings[t]
                ids.append(idx)
                tfs.append(tf)
    N = len(docs)
    dfs = {t: len(ids) for t, (ids, _tfs) in postings.items()}
    idfs = {t: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for t, d in dfs.items()}
    return _LightBM25Index(terms, docs, dls, postings, idfs)


_LIGHT_GLOBAL_CACHE: dict[str, _LightBM25Index] = {}


def light_global_index(rows: list[dict]) -> _LightBM25Index:
    key = f"{len(rows)}:{hash(tuple((r.get('question_text') or r.get('question') or '') for r in rows))}"
    if key not in _LIGHT_GLOBAL_CACHE:
        _LIGHT_GLOBAL_CACHE[key] = _light_global_index(rows)
    return _LIGHT_GLOBAL_CACHE[key]


def _ranked_abstain_pages(row: dict, top_k: int = 8) -> list[dict]:
    registry = load_company_registry()
    res = resolve_query(row.get("question_text") or row.get("question") or "", registry)
    chunks = load_page_chunks()
    frow = dict(row)
    if res.company_ids:
        frow["company_id"] = "|".join(res.company_ids)
    if res.periods:
        frow["fiscal_year"] = res.periods[0].token
    if res.basis:
        frow["variant"] = res.basis
    cands = _filter_chunks(chunks, frow)
    if not cands:
        cands = _filter_chunks(chunks, {"company_id": frow.get("company_id")})
    if not cands:
        return []
    key = (tuple(sorted(str(row.get("company_id") or "").split("|"))),
           row.get("fiscal_year") or row.get("period"), row.get("variant"))
    if key not in _ABSTAIN_CACHE:
        _ABSTAIN_CACHE[key] = BM25Index(cands)
    idx = _ABSTAIN_CACHE[key]
    ranked = idx.search(row.get("question_text") or row.get("question") or "", k=top_k)
    # _filter_chunks uses the same candidates for this key, so indices align.

    pages = []
    for i, _score in ranked:
        c = cands[i]
        m = c["meta"]
        pages.append({"document_id": m["document_id"], "page": m["page"],
                      "text": c["text"], "rank": len(pages) + 1})
    return pages


_RAG_CACHE: dict[tuple, BM25Index] = {}
_RAG_CANDIDATE_CACHE: dict[tuple, list[dict]] = {}


def _rag_filtered_candidates(row: dict) -> list[dict]:
    registry = load_company_registry()
    res = resolve_query(row.get("question_text") or row.get("question") or "", registry)
    chunks = load_page_chunks()
    frow = dict(row)
    if res.company_ids:
        frow["company_id"] = "|".join(res.company_ids)
    if res.periods:
        frow["fiscal_year"] = res.periods[0].token
    if res.basis:
        frow["variant"] = res.basis
    cands = _filter_chunks(chunks, frow)
    if not cands:
        cands = _filter_chunks(chunks, {"company_id": frow.get("company_id")})
    if not cands:
        return []
    return cands


def _rag_candidates(row: dict, filtered: bool) -> list[dict]:
    if not filtered:
        from src.eval.prompting.inputs import load_dev_rows
        idx = light_global_index(load_dev_rows())
        ranked = idx.search(row.get("question_text") or row.get("question") or "", k=8)
        pages = []
        for i, _score in ranked:
            did, page = idx.docs[i]
            text = _page_text(did, page)
            if text:
                pages.append({"document_id": did, "page": page, "text": text,
                              "rank": len(pages) + 1})
        return pages
    cands = _rag_filtered_candidates(row)
    if not cands:
        return []
    key = ("filtered", tuple(sorted(str(row.get("company_id") or "").split("|"))),
           row.get("fiscal_year") or row.get("period"), row.get("variant"))
    idx = _RAG_CACHE.get(key)
    if idx is None or idx.N != len(cands):
        idx = BM25Index(cands)
        _RAG_CACHE[key] = idx
    ranked = idx.search(row.get("question_text") or row.get("question") or "", k=8)
    pages = []
    for i, _score in ranked:
        if i < 0 or i >= len(cands):
            continue
        c = cands[i]
        m = c["meta"]
        pages.append({"document_id": m["document_id"], "page": m["page"],
                      "text": c["text"], "rank": len(pages) + 1})
    return pages


def rag_candidate_pages(row: dict, *, filtered: bool = False, top_k: int = 8) -> list[dict]:
    key = ("filtered" if filtered else "global", row.get("id") or row.get("row_id"))
    # Cache candidate pages separately from the BM25 index.
    cache_key = (key[0], row.get("id") or row.get("row_id"))
    if cache_key not in _RAG_CANDIDATE_CACHE:
        _RAG_CANDIDATE_CACHE[cache_key] = _rag_candidates(row, filtered)[:top_k]
    return [dict(p) for p in _RAG_CANDIDATE_CACHE[cache_key]]


def _gold_hash() -> str:
    from src.config import GOLD_PATH
    return hashlib.sha256(Path(GOLD_PATH).read_bytes()).hexdigest()


def _corpus_freeze_sha() -> str:
    return json.loads(Path("data/corpus/freeze_v1.json").read_text(encoding="utf-8"))["manifest_sha256"]


def _context_cache_path() -> Path:
    return Path("data/eval/prompts") / f"context_cache_{_corpus_freeze_sha()}.jsonl"


def load_context_cache() -> dict[str, list[dict]]:
    global _CONTEXT_CACHE, _CONTEXT_CACHE_META
    if _CONTEXT_CACHE is not None:
        return _CONTEXT_CACHE
    path = _context_cache_path()
    if not path.is_file():
        raise FileNotFoundError(f"context cache not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty context cache: {path}")
    meta = rows[0]
    if meta.get("gold_hash") != _gold_hash():
        raise ValueError("context cache gold hash mismatch")
    if meta.get("corpus_freeze_sha") != _corpus_freeze_sha():
        raise ValueError("context cache corpus freeze sha mismatch")
    cache = {}
    for r in rows[1:]:
        if r.get("gold_hash") != _gold_hash() or r.get("corpus_freeze_sha") != _corpus_freeze_sha():
            raise ValueError(f"context cache metadata mismatch for {r.get('cache_id')}")
        cache[r["cache_id"]] = r["candidate_pages"]
    _CONTEXT_CACHE = cache
    _CONTEXT_CACHE_META = {"gold_hash": _gold_hash(), "corpus_freeze_sha": _corpus_freeze_sha(),
                          "path": str(path), "n_rows": len(cache)}
    return cache


def candidate_pages(row: dict) -> list[dict]:
    """Cached candidate pages; never rebuild BM25 per call."""
    cache = load_context_cache()
    rid = row.get("id") or row.get("row_id")
    if rid not in cache:
        raise KeyError(f"row {rid} missing from context cache")
    return [dict(p) for p in cache[rid]]


def _compute_candidate_pages(row: dict) -> list[dict]:
    """Rank-ordered pages for B1 context, before packing."""
    if row.get("expected_behaviour") != "abstain":
        pages = []
        for e in row.get("evidence") or []:
            did = e.get("document_id")
            pg = e.get("pdf_page")
            if not did or not isinstance(pg, int):
                continue
            text = _page_text(did, pg)
            if text:
                pages.append({"document_id": did, "page": pg, "text": text,
                              "rank": len(pages) + 1})
        if pages:
            return pages
    return _ranked_abstain_pages(row)


def pack_pages(pages: list[dict], system: str, question: str, max_tokens: int,
               ceiling: int) -> tuple[str, set[tuple[str, int]], bool]:
    """Pack highest-ranked pages until the prompt fits. Returns (context, keys, ok)."""
    from src.llm.client import _estimate_tokens

    def render(selected):
        blocks = [f"[document_id={p['document_id']} page={p['page']}]\n{p['text']}"
                  for p in selected]
        return "\n\n".join(blocks)

    safety = min(1024, max(0, (ceiling - max_tokens) // 10))

    def fits(text: str) -> bool:
        prompt = f"{system}\n\n{text}\n\nQuestion: {question}"
        return _estimate_tokens(prompt) + max_tokens <= ceiling - safety

    selected = list(pages)
    while selected:
        context = render(selected)
        if fits(context):
            keys = {(p["document_id"], p["page"]) for p in selected}
            return context, keys, True
        selected.pop()

    # No pages fit; try one minimal/truncated chunk if any page exists.
    if pages:
        p = pages[0]
        header = f"[document_id={p['document_id']} page={p['page']}]\n"
        text = p.get("text") or ""
        # Binary-search truncation on characters.
        lo, hi, best = 0, len(text), None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = header + text[:mid]
            if fits(candidate):
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            return best, {(p["document_id"], p["page"])}, True
        return "", set(), False

    # No context pages at all.
    if fits(""):
        return "", set(), True
    return "", set(), False
