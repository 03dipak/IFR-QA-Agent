from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import operator
import os
import pickle
import re
import threading
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src import request_deadline as budget

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_META_CACHE_DIR = Path(os.environ.get("AGENT_META_CACHE_DIR", "data/cache"))
LOGGER = logging.getLogger(__name__)
_RERANK_MODEL: Any | None = None
_RERANK_LOCK = threading.Lock()
_QUERY_EMBED_CACHE: dict[tuple[str, str], np.ndarray] = {}
_QUERY_EMBED_LOCK = threading.Lock()
_WARMUP_QUERY = "annual report revenue"


def _cache_enabled() -> bool:
    value = os.environ.get("AGENT_CACHE_ENABLED")
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _retrieval_config() -> dict[str, Any]:
    path = Path("configs/retrieval.yaml")
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _hybrid_enabled() -> bool:
    override = os.environ.get("RETRIEVAL_HYBRID")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return bool((_retrieval_config().get("retrieval") or {}).get("hybrid", False))


def _hybrid_candidate_k() -> int:
    override = os.environ.get("RETRIEVAL_HYBRID_CANDIDATE_K")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    config = _retrieval_config().get("retrieval") or {}
    return max(1, int(config.get("candidate_k", 1000)))


def _hybrid_rrf_k() -> int:
    return max(1, int((_retrieval_config().get("hybrid") or {}).get("rrf_k", 60)))


def _hybrid_pool_k() -> int:
    return max(1, int(
        (_retrieval_config().get("hybrid") or {}).get("candidate_pool_k", 30)
    ))


def _rerank_pool_k() -> int:
    override = os.environ.get("RETRIEVAL_RERANK_POOL")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max(1, int(
        (_retrieval_config().get("rerank") or {}).get("candidate_pool_k", 30)
    ))


def _rerank_enabled() -> bool:
    override = os.environ.get("RETRIEVAL_RERANK")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return bool((_retrieval_config().get("rerank") or {}).get("enabled", True))


def _rerank_model() -> Any:
    if not _rerank_enabled():
        raise RuntimeError("reranker disabled by RETRIEVAL_RERANK; Track B requires it")
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        with _RERANK_LOCK:
            if _RERANK_MODEL is None:
                model_name = os.environ.get("RERANK_MODEL")
                if not model_name:
                    raise RuntimeError("RERANK_MODEL is not configured")
                cache_dir_value = os.environ.get("RERANK_CACHE_DIR")
                if not cache_dir_value:
                    raise RuntimeError("RERANK_CACHE_DIR is not configured")
                cache_dir = Path(cache_dir_value)
                if not cache_dir.is_absolute():
                    cache_dir = PROJECT_ROOT / cache_dir
                started = time.monotonic()
                LOGGER.info(
                    "rerank_load_start model=%s cache=%s", model_name, cache_dir.name
                )
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                    threads = os.environ.get("RERANK_THREADS")
                    if not threads:
                        raise RuntimeError("RERANK_THREADS is not configured")
                    _RERANK_MODEL = TextCrossEncoder(
                        model_name=model_name, cache_dir=str(cache_dir), threads=int(threads)
                    )
                except Exception as exc:
                    LOGGER.error(
                        "rerank_load_failed error=%s latency_s=%.3f",
                        type(exc).__name__, time.monotonic() - started,
                    )
                    raise RuntimeError(
                        f"reranker unavailable: {type(exc).__name__}: {exc}"
                    ) from exc
                LOGGER.info(
                    "rerank_load_end latency_s=%.3f", time.monotonic() - started
                )
    return _RERANK_MODEL


def _index_path() -> Path:
    value = os.environ.get("AGENT_INDEX_PATH")
    if not value:
        raise RuntimeError("AGENT_INDEX_PATH is not configured")
    return Path(value)


def _resolved_index_path(path: str | None = None) -> str:
    candidate = Path(path) if path else _index_path()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate.resolve())


@lru_cache(maxsize=4)
def _open_index_resolved(resolved_path: str) -> Any:
    from src.rag_ingestion.full_index import FullVectorIndex

    return FullVectorIndex(Path(resolved_path))


def _open_index(path: str | None = None) -> Any:
    started = time.monotonic()
    LOGGER.info("index_load_start")
    index = _open_index_resolved(_resolved_index_path(path))
    LOGGER.info(
        "index_load_end latency_s=%.3f cached=%s",
        time.monotonic() - started,
        _open_index_resolved.cache_info().hits > 0,
    )
    return index


def warm_index() -> str:
    index = _open_index()
    return str(getattr(index, "backend", "faiss"))


def index_backend(path: str | None = None) -> str:
    return str(getattr(_open_index(path), "backend", "faiss"))


@lru_cache(maxsize=4)
def _index_meta(resolved_path: str) -> dict[str, Any]:
    return json.loads((Path(resolved_path) / "meta.json").read_text(encoding="utf-8"))


def _index_meta_for(path: str | None = None) -> dict[str, Any]:
    return _index_meta(_resolved_index_path(path))


def _ids_from_meta(meta: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("company_ids", "companies", "counts_by_company"):
        value = meta.get(key)
        if isinstance(value, dict):
            candidates = list(value.keys())
        elif isinstance(value, (list, tuple)):
            candidates = list(value)
        else:
            continue
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                ids.add(candidate.strip())
    return ids


@lru_cache(maxsize=4)
def company_ids(path: str | None = None) -> tuple[str, ...]:
    resolved_path = _resolved_index_path(path)
    root = Path(resolved_path)
    ids_path = root / "company_ids.json"
    if ids_path.is_file():
        payload = json.loads(ids_path.read_text(encoding="utf-8"))
        values = payload.get("company_ids") if isinstance(payload, dict) else payload
        if isinstance(values, (list, tuple)) and any(isinstance(v, str) and v.strip() for v in values):
            return tuple(sorted(
                value.strip() for value in values
                if isinstance(value, str) and value.strip()
            ))
    ids = _ids_from_meta(_index_meta(resolved_path))
    if not ids:
        try:
            ids = {row[0] for row in _meta_rows_resolved(resolved_path) if row and row[0]}
        except Exception:
            ids = set()
    ids_path.write_text(
        json.dumps({"company_ids": sorted(ids)}, indent=2), encoding="utf-8"
    )
    return tuple(sorted(ids))


def _source_filename(meta: Mapping[str, Any]) -> str:
    for key in ("source", "filename", "source_path", "document", "doc_id"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value.lower()
    return ""


def _meta_row(meta: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    company = meta.get("company_id") or meta.get("company") or ""
    period = meta.get("period") or ""
    statement_kind = meta.get("statement_kind") or ""
    doc_type = meta.get("doc_type") or ""
    reporting_basis = meta.get("reporting_basis") or ""
    return (
        str(company).strip(),
        str(period).strip(),
        str(statement_kind).strip(),
        str(doc_type).strip(),
        _source_filename(meta),
        str(reporting_basis).strip(),
    )


def _meta_cache_path(root: Path, meta: Mapping[str, Any]) -> Path:
    digest = str(meta.get("chunk_ids_sha256") or "unknown")
    count = str(meta.get("chunks") or "0")
    stat = (root / "chunks.jsonl").stat()
    return _META_CACHE_DIR / (
        f"meta_rows_{digest[:16]}_{count}_{stat.st_size}_{stat.st_mtime_ns}.pkl"
    )


@lru_cache(maxsize=4)
def _meta_rows_resolved(resolved_path: str) -> tuple[tuple[str, str, str, str, str, str], ...]:
    root = Path(resolved_path)
    ids = json.loads((root / "chunk_ids.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    cache = _meta_cache_path(root, meta)
    if cache.is_file() and _cache_enabled():
        try:
            with cache.open("rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, tuple) and len(cached) == len(ids):
                return cached
        except Exception:
            LOGGER.warning("ignored unusable meta cache: %s", cache.name)
    started = time.monotonic()
    rows: dict[str, tuple[str, str, str, str, str, str]] = {}
    with (root / "chunks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                rows[str(chunk_id)] = _meta_row(chunk.get("meta") or {})
    empty = _meta_row({})
    resolved = tuple(rows.get(str(chunk_id), empty) for chunk_id in ids)
    try:
        if not _cache_enabled():
            raise OSError("cache disabled")
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(".tmp")
        with temp.open("wb") as handle:
            pickle.dump(resolved, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(cache)
    except Exception:
        LOGGER.warning("could not write meta cache: %s", cache.name)
    LOGGER.info(
        "meta_rows_built rows=%d latency_s=%.3f cache=%s",
        len(resolved), time.monotonic() - started, cache.name,
    )
    return resolved


@lru_cache(maxsize=4)
def _chunk_ids_by_company_period(resolved_path: str) -> dict[tuple[str, str], tuple[str, ...]]:
    ids = json.loads((Path(resolved_path) / "chunk_ids.json").read_text(encoding="utf-8"))
    rows = _meta_rows_resolved(resolved_path)
    out: dict[tuple[str, str], list[str]] = {}
    for chunk_id, row in zip(ids, rows, strict=False):
        company = str(row[0] or "").strip().lower()
        period = str(row[1] or "").strip().upper()
        if not company:
            continue
        key = (company, period)
        out.setdefault(key, []).append(str(chunk_id))
    return {k: tuple(v) for k, v in out.items()}


def _meta_rows_for_index(index: Any, path: str) -> tuple[tuple[str, str, str, str, str, str], ...]:
    supplied = getattr(index, "chunk_metas", None)
    if supplied is not None:
        return tuple(_meta_row(meta or {}) for meta in supplied)
    return _meta_rows_resolved(_resolved_index_path(path))


def _query_embedding(query: str, index: Any, path: str) -> np.ndarray:
    from src.rag_ingestion.full_index import embed_text

    meta = getattr(index, "meta", None) or _index_meta_for(path)
    embedder = getattr(index, "embedder", None) or meta.get("embedder")
    if embedder is None and str(meta.get("model", "")).startswith("local:hash"):
        embedder = "hash384"
    embedder = embedder or "hash384"
    dim = int(meta.get("dim", 384))
    query_text = query
    if embedder == "bge-small":
        from src.rag_ingestion.full_index import BGE_QUERY_PREFIX
        query_text = BGE_QUERY_PREFIX + query
    model = str(meta.get("model") or embedder)
    key = (model, query_text)
    with _QUERY_EMBED_LOCK:
        cached = _QUERY_EMBED_CACHE.get(key) if _cache_enabled() else None
    if cached is not None:
        return cached
    started = time.monotonic()
    LOGGER.info("query_embedding_start embedder=%s", embedder)
    vector = np.asarray(embed_text(query_text, dim, embedder), dtype=np.float32)
    LOGGER.info(
        "query_embedding_end latency_s=%.3f cache_hit=false",
        time.monotonic() - started,
    )
    if _cache_enabled():
        with _QUERY_EMBED_LOCK:
            _QUERY_EMBED_CACHE[key] = vector
    return vector


def warm_meta_rows() -> int:
    started = time.monotonic()
    LOGGER.info("meta_rows_warm_start")
    rows = _meta_rows_resolved(_resolved_index_path())
    LOGGER.info(
        "meta_rows_warm_end rows=%d latency_s=%.3f", len(rows), time.monotonic() - started
    )
    return len(rows)


def warm_reranker() -> bool:
    _rerank_model()
    return True


def warm_query_embedding(timeout_s: float | None = None) -> float:
    from concurrent.futures import ThreadPoolExecutor

    if timeout_s is None:
        value = os.environ.get("QUERY_EMBED_WARM_TIMEOUT_SECONDS")
        if not value:
            raise RuntimeError("QUERY_EMBED_WARM_TIMEOUT_SECONDS is not configured")
        timeout_s = float(value)
    started = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_query_embedding, _WARMUP_QUERY, _open_index(), "")
        future.result(timeout=timeout_s)
    except TimeoutError as exc:
        elapsed = time.monotonic() - started
        LOGGER.warning("query embedding warmup timed out after %.3fs", elapsed)
        raise TimeoutError(f"query embedding warmup exceeded {timeout_s:g}s") from exc
    finally:
        executor.shutdown(wait=False)
    latency = time.monotonic() - started
    LOGGER.info("query embedding warmup latency_s=%.3f query=%s", latency, _WARMUP_QUERY)
    return latency


_GENERIC_COMPANY_TOKENS = {"green", "pharma", "steel", "sun"}
_COMPANY_STOPWORDS = {
    "about", "after", "against", "and", "annual", "balance", "between", "cash",
    "compare", "compared", "company", "companies", "consolidated", "crore",
    "difference", "ended", "equity", "figure", "figures", "financial", "flow",
    "for", "from", "higher", "how", "income", "larger", "many", "march",
    "million", "much", "operations", "period", "printed", "profit", "quarter",
    "reading", "revenue", "standalone", "statement", "statements", "the", "their", "total",
    "versus", "what", "which", "with", "would", "year",
} | _GENERIC_COMPANY_TOKENS


def _id_tokens(company: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", company.lower()) if token]


def _company_alias_tokens(company_id: str) -> set[str]:
    return {
        token
        for token in _id_tokens(company_id)
        if len(token) >= 3 and token not in _COMPANY_STOPWORDS
    }


def _unique_alias_map(company_ids_list: list[str]) -> dict[str, set[str]]:
    counts: dict[str, int] = {}
    for company in company_ids_list:
        for token in _company_alias_tokens(company):
            counts[token] = counts.get(token, 0) + 1
    return {
        company: {
            token
            for token in _company_alias_tokens(company)
            if counts.get(token) == 1
        }
        for company in company_ids_list
    }


def _token_matches_alias(token: str, aliases: set[str]) -> bool:
    if token in aliases:
        return True
    if token + "s" in aliases:
        return True
    if token.endswith("s") and token[:-1] in aliases:
        return True
    return False


def detect_companies(
    query: str,
    ids: list[str] | tuple[str, ...] | None = None,
    unresolved: list[str] | None = None,
    logged_unresolved: set[str] | None = None,
) -> list[str]:
    available = list(ids) if ids is not None else list(company_ids())
    cleaned = re.sub(r"[^a-z0-9\s]", " ", query.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    query_tokens = [token for token in cleaned.split() if token]
    matches: set[str] = set()
    unresolved_names: set[str] = set()

    aliases: dict[str, set[str]] = _unique_alias_map(available)
    for token in query_tokens:
        if len(token) < 3 or token in _COMPANY_STOPWORDS:
            continue
        if re.fullmatch(r"fy\d{4}", token) or re.fullmatch(r"\d{4}", token):
            continue
        targets = [
            company
            for company, tokens in aliases.items()
            if _token_matches_alias(token, tokens)
        ]
        if targets:
            matches.update(targets)

    for left, right in zip(query_tokens, query_tokens[1:]):
        if (
            len(left) < 3 or len(right) < 3
            or left in _COMPANY_STOPWORDS or right in _COMPANY_STOPWORDS
        ):
            continue
        pair_targets = [
            company
            for company, tokens in aliases.items()
            if _token_matches_alias(left, tokens) and _token_matches_alias(right, tokens)
        ]
        if pair_targets:
            matches.update(pair_targets)

    proper = {name.lower() for name in re.findall(r"\b[A-Z][A-Za-z0-9]+\b", query)}
    for name in sorted(proper):
        if len(name) < 4 or name in _COMPANY_STOPWORDS:
            continue
        if any(_token_matches_alias(name, tokens) for tokens in aliases.values()):
            continue
        unresolved_names.add(name)

    for name in sorted(unresolved_names):
        if logged_unresolved is None or name not in logged_unresolved:
            LOGGER.info("unresolved_company_alias: %s", name)
            if logged_unresolved is not None:
                logged_unresolved.add(name)
        if unresolved is not None:
            unresolved.append(name)
    return sorted(matches)


def _company_match(meta: Mapping[str, Any], company: str) -> str | None:
    needle = company.strip().lower()
    if not needle:
        return None
    exact = str(meta.get("company_id", "")).strip().lower()
    if exact == needle:
        return "exact"
    if needle in exact or needle in _source_filename(meta):
        return "substring_fallback"
    return None


def _meta_period(meta: Mapping[str, Any]) -> str:
    value = meta.get("period")
    return str(value).strip() if value not in (None, "") else ""


def _merge_info(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if key == "excluded_empty_period":
            base[key] = int(base.get(key, 0)) + int(value)
        else:
            base[key] = value


def _normalize_basis(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"consolidated", "standalone"}:
        return text
    return None


def _row_matches(
    row: tuple[str, str, str, str, str, str],
    company: str | None,
    period: str | None,
    statement_kind: str | None,
    doc_type: str | None,
    reporting_basis: str | None = None,
) -> bool:
    actual_company, actual_period, actual_kind, actual_type, _, actual_basis = row
    if company and actual_company.lower() != company.strip().lower():
        return False
    if period and not actual_period.upper().startswith(period.strip().upper()):
        return False
    if statement_kind and actual_kind.lower() != statement_kind.strip().lower():
        return False
    if doc_type and actual_type.lower() != doc_type.strip().lower():
        return False
    requested_basis = _normalize_basis(reporting_basis)
    if requested_basis:
        if _normalize_basis(actual_basis) != requested_basis:
            return False
    return True


def _hybrid_hits(
    query: str,
    k: int,
    index: Any,
    positions: np.ndarray,
    scores: np.ndarray,
    match_info: dict[str, Any],
    filtered: bool = False,
    rerank_pool: int | None = None,
) -> list[dict[str, Any]]:
    from rank_bm25 import BM25Okapi

    from src.tools.page_retrieve import _rrf, cleaned_tokens

    if positions.size > _hybrid_candidate_k():
        top = np.argpartition(-scores, _hybrid_candidate_k() - 1)[:_hybrid_candidate_k()]
        positions = positions[top]
        scores = scores[top]
    order = np.argsort(-scores, kind="stable")
    positions = positions[order]
    scores = scores[order]
    chunks = [index._chunk_at(int(position)) for position in positions]
    texts = [str(chunk.get("text") or "") for chunk in chunks]
    tokenized = [cleaned_tokens(text) for text in texts]
    dense_rank = [(rank, float(score)) for rank, score in enumerate(scores)]
    if any(tokenized):
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(cleaned_tokens(query))
        bm25_order = np.argsort(-bm25_scores, kind="stable")
        bm25_rank = [(int(position), float(bm25_scores[position]))
                     for position in bm25_order]
    else:
        bm25_rank = list(dense_rank)
    pool_k = max(1, int(rerank_pool if rerank_pool is not None else _rerank_pool_k()))
    pool = _rrf([bm25_rank, dense_rank], _hybrid_rrf_k(), max(_hybrid_pool_k(), pool_k))
    pool = pool[:pool_k]
    model = _rerank_model()
    rerank_scores: list[float]
    if model is not None:
        try:
            rerank_scores = [
                float(score) for score in model.rerank(query, [texts[i] for i, _ in pool])
            ]
        except Exception:
            rerank_scores = [score for _, score in pool]
    else:
        rerank_scores = [score for _, score in pool]
    # soft boost for P&L metrics: prefer income_statement, no hard filter
    ql = str(query or "").lower()
    pl_query = any(t in ql for t in ("revenue", "profit", "income", "expense", "eps", "ebitda", "margin"))
    bs_query = any(t in ql for t in ("total assets", "total liabilities", "equity", "balance sheet"))
    boost = pl_query and not bs_query
    ordered = sorted(
        zip([position for position, _ in pool], rerank_scores),
        key=lambda item: (
            -float(item[1]) - (0.25 if boost and str((chunks[int(item[0])].get("meta") or {}).get("statement_kind") or "").lower() == "income_statement" else 0.0),
        ),
    )
    match_info["retrieval_mode"] = "hybrid"
    match_info["hybrid_candidates"] = int(positions.size)
    match_info["hybrid_pool"] = len(pool)
    hits: list[dict[str, Any]] = []
    for local_position, score in ordered[:k]:
        chunk = dict(chunks[int(local_position)])
        chunk["chunk_id"] = str(index.ids[int(positions[int(local_position)])])
        chunk["score"] = float(score)
        chunk["match_mode"] = "exact" if filtered else "hybrid"
        hits.append(chunk)
    return hits


def _search_once(
    query: str,
    k: int,
    company: str | None,
    period: str | None,
    statement_kind: str | None,
    doc_type: str | None,
    reporting_basis: str | None = None,
    index: Any | None = None,
    query_vector: np.ndarray | None = None,
    hybrid: bool = False,
    rerank_pool: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _resolved_index_path()
    index = index if index is not None else _open_index(path)
    requested_basis = _normalize_basis(reporting_basis)
    filtered = bool(company or period or statement_kind or doc_type or requested_basis)
    match_info: dict[str, Any] = {
        "excluded_empty_period": 0,
        "filter_mode": "masked" if filtered else "faiss",
    }
    if company:
        match_info["company_filter"] = company
    if period:
        match_info["period_filter"] = period
    if statement_kind:
        match_info["statement_kind_filter"] = statement_kind
    if doc_type:
        match_info["doc_type_filter"] = doc_type
    if requested_basis:
        match_info["reporting_basis_filter"] = requested_basis
    if not filtered:
        if hybrid:
            q = query_vector if query_vector is not None else _query_embedding(
                query, index, path
            )
            scores = np.asarray(
                np.asarray(index.embeddings) @ np.asarray(q, dtype=np.float32)
            ).reshape(-1)
            positions = np.arange(scores.size)
            return _hybrid_hits(query, k, index, positions, scores, match_info, filtered=False, rerank_pool=rerank_pool), match_info
        return [dict(hit) for hit in index.search(query, k)], match_info

    q = query_vector if query_vector is not None else _query_embedding(query, index, path)
    q = np.asarray(q, dtype=np.float32)
    rows = _meta_rows_for_index(index, path)
    mask = np.zeros(len(rows), dtype=bool)
    for position, row in enumerate(rows):
        if period and not row[1]:
            match_info["excluded_empty_period"] += 1
            continue
        mask[position] = _row_matches(
            row, company, period, statement_kind, doc_type, requested_basis
        )
    positions = np.flatnonzero(mask)
    match_info["masked_rows"] = int(positions.size)
    if positions.size == 0:
        return [], match_info

    embeddings = np.asarray(index.embeddings)[positions]
    scores = np.asarray(embeddings @ q).reshape(-1)
    if hybrid:
        return _hybrid_hits(query, k, index, positions, scores, match_info, filtered=True, rerank_pool=rerank_pool), match_info
    if scores.size > k:
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
    else:
        top = np.argsort(-scores)
    hits: list[dict[str, Any]] = []
    for offset in top:
        position = int(positions[int(offset)])
        hit = dict(index._chunk_at(position))
        hit["chunk_id"] = str(index.ids[position])
        hit["score"] = float(scores[int(offset)])
        hit["match_mode"] = "exact"
        hits.append(hit)
    return hits, match_info


_SEARCH_CALL_COUNT = 0
_SEARCH_CACHE_LOCK = threading.Lock()
_SEARCH_CACHE_SCHEMA = "search_index_v3_header_extraction"


def _search_cache_dir() -> Path:
    value = os.environ.get("AGENT_SEARCH_CACHE_DIR", "data/cache/search_index")
    cache_dir = Path(value)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    return cache_dir


@lru_cache(maxsize=4)
def _search_index_fingerprint(resolved_path: str) -> dict[str, Any]:
    root = Path(resolved_path)
    meta = _index_meta(resolved_path)
    chunks = root / "chunks.jsonl"
    try:
        stat = chunks.stat()
        chunk_stat = [stat.st_size, stat.st_mtime_ns]
    except OSError:
        chunk_stat = None
    return {
        "path": str(root),
        "chunk_ids_sha256": str(meta.get("chunk_ids_sha256") or ""),
        "chunks": str(meta.get("chunks") or ""),
        "chunk_stat": chunk_stat,
    }


def _search_settings_fingerprint(rerank_pool: int | None = None) -> dict[str, Any]:
    return {
        "schema": _SEARCH_CACHE_SCHEMA,
        "hybrid": _hybrid_enabled(),
        "candidate_k": _hybrid_candidate_k(),
        "rrf_k": _hybrid_rrf_k(),
        "hybrid_pool_k": _hybrid_pool_k(),
        "rerank": _rerank_enabled(),
        "rerank_pool_k": int(rerank_pool) if rerank_pool is not None else _rerank_pool_k(),
        "rerank_model": os.environ.get("RERANK_MODEL"),
        "rerank_cache_dir": os.environ.get("RERANK_CACHE_DIR"),
        "rerank_threads": os.environ.get("RERANK_THREADS"),
    }


def _search_cache_key(
    query: str,
    k: int,
    company: str | None,
    period: str | None,
    statement_kind: str | None,
    doc_type: str | None,
    reporting_basis: str | None,
    resolved_path: str,
    rerank_pool: int | None = None,
) -> str:
    payload = {
        "query": str(query),
        "k": int(k),
        "company": company,
        "period": period,
        "statement_kind": statement_kind,
        "doc_type": doc_type,
        "reporting_basis": reporting_basis,
        "index": _search_index_fingerprint(resolved_path),
        "settings": _search_settings_fingerprint(rerank_pool),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_search_cache(cache_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "hits" not in payload or "match_info" not in payload:
        return None
    return copy.deepcopy(payload)


def _write_search_cache(cache_path: Path, result: dict[str, Any]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
        temp.write_text(json.dumps(result, default=str), encoding="utf-8")
        with _SEARCH_CACHE_LOCK:
            temp.replace(cache_path)
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("search_cache_write_failed error=%s", type(exc).__name__)


def search_index(
    query: str,
    k: int = 5,
    company: str | None = None,
    period: str | None = None,
    statement_kind: str | None = None,
    doc_type: str | None = None,
    reporting_basis: str | None = None,
    rerank_pool: int | None = None,
) -> dict[str, Any]:
    global _SEARCH_CALL_COUNT
    _SEARCH_CALL_COUNT += 1
    call_number = _SEARCH_CALL_COUNT
    started = time.monotonic()
    filtered = bool(company or period or statement_kind or doc_type or reporting_basis)
    LOGGER.info(
        "search_start call=%d query=%r company=%s period=%s filtered=%s",
        call_number, query, company, period, filtered,
    )
    try:
        if k <= 0:
            return {"hits": [], "match_info": {"excluded_empty_period": 0}}
        requested_basis = _normalize_basis(reporting_basis)
        path = _resolved_index_path()
        cache_path = None
        if _cache_enabled():
            cache_path = _search_cache_dir() / (
                _search_cache_key(
                    query, k, company, period, statement_kind, doc_type,
                    reporting_basis, path, rerank_pool=rerank_pool,
                ) + ".json"
            )
            cached = _read_search_cache(cache_path)
            if cached is not None:
                LOGGER.info("search_cache_hit call=%d path=%s", call_number, cache_path.name)
                return cached
        index = _open_index(path)
        hybrid = _hybrid_enabled()
        has_filters = bool(
            company or period or statement_kind or doc_type or requested_basis
        )
        query_vector = (
            _query_embedding(query, index, path) if has_filters or hybrid else None
        )
        relaxed: list[str] = []
        cur_period, cur_sk, cur_dt, cur_basis = (
            period, statement_kind, doc_type, requested_basis
        )
        hits, match_info = _search_once(
            query, k, company, cur_period, cur_sk, cur_dt, cur_basis,
            index, query_vector, hybrid, rerank_pool=rerank_pool,
        )
        if not hits and cur_sk:
            relaxed.append("drop_statement_kind")
            cur_sk = None
            hits, info = _search_once(
                query, k, company, cur_period, cur_sk, cur_dt, cur_basis,
                index, query_vector, hybrid, rerank_pool=rerank_pool,
            )
            _merge_info(match_info, info)
        if not hits and cur_basis:
            relaxed.append("drop_reporting_basis")
            cur_basis = None
            hits, info = _search_once(
                query, k, company, cur_period, cur_sk, cur_dt, cur_basis,
                index, query_vector, hybrid, rerank_pool=rerank_pool,
            )
            _merge_info(match_info, info)
        if not hits and cur_period:
            relaxed.append("drop_period")
            cur_period = None
            hits, info = _search_once(
                query, k, company, cur_period, cur_sk, cur_dt, cur_basis,
                index, query_vector, hybrid, rerank_pool=rerank_pool,
            )
            _merge_info(match_info, info)
        match_info["relaxed"] = relaxed
        match_info["final_period_filter"] = cur_period
        match_info["final_statement_kind_filter"] = cur_sk
        match_info["final_doc_type_filter"] = cur_dt
        match_info["final_reporting_basis_filter"] = cur_basis
        result = {"hits": hits, "match_info": match_info}
        if cache_path is not None:
            _write_search_cache(cache_path, result)
        return result
    finally:
        LOGGER.info(
            "search_index call=%d first=%s elapsed=%.3fs filtered=%s",
            call_number,
            call_number == 1,
            time.monotonic() - started,
            filtered,
        )


@lru_cache(maxsize=4)
def _chunk_offsets_resolved(resolved_path: str) -> dict[str, tuple[int, int]]:
    root = Path(resolved_path)
    ids = json.loads((root / "chunk_ids.json").read_text(encoding="utf-8"))
    raw = json.loads((root / "chunk_offsets.json").read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return {chunk_id: (int(raw[chunk_id][0]), int(raw[chunk_id][1]))
                for chunk_id in ids}
    if len(ids) != len(raw):
        raise ValueError("chunk ids and offsets disagree")
    return {chunk_id: (int(offset[0]), int(offset[1]))
            for chunk_id, offset in zip(ids, raw, strict=True)}


@lru_cache(maxsize=4)
def _page_chunk_map_resolved(resolved_path: str) -> dict[tuple[str, str, int], str]:
    rows = _meta_rows_resolved(resolved_path)
    chunk_ids = json.loads((Path(resolved_path) / "chunk_ids.json").read_text(encoding="utf-8"))
    page_map: dict[tuple[str, str, int], str] = {}
    for chunk_id, row in zip(chunk_ids, rows, strict=False):
        cid = str(chunk_id).strip()
        if ":p" not in cid:
            continue
        company = str(row[0]).strip().lower()
        document_id = str(row[4]).strip()
        page = row[5]
        if not company or not document_id or page in (None, ""):
            continue
        try:
            key = (company, document_id, int(page))
        except (TypeError, ValueError):
            continue
        page_map.setdefault(key, cid)
    return page_map


def get_chunk(chunk_id: str) -> dict[str, Any]:
    budget.check()
    root = Path(_resolved_index_path())
    offset, size = _chunk_offsets_resolved(str(root))[chunk_id]
    with (root / "chunks.jsonl").open("rb") as fh:
        fh.seek(offset)
        return json.loads(fh.read(size).decode("utf-8"))


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_arithmetic(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_arithmetic(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_arithmetic(node.left)
        right = _eval_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def _normalize_calculator_expression(expression: str) -> str:
    normalized = str(expression).replace("\u2212", "-")
    cleaned: list[str] = []
    for index, char in enumerate(normalized):
        if char in ", ":
            prev = normalized[index - 1] if index > 0 else ""
            nxt = normalized[index + 1] if index + 1 < len(normalized) else ""
            if char == " " or (char == "," and prev.isdigit() and nxt.isdigit()):
                continue
        cleaned.append(char)
    return "".join(cleaned)


def calculator(expression: str) -> int | float:
    if len(expression) > 200:
        raise ValueError("expression too long")
    tree = ast.parse(_normalize_calculator_expression(expression), mode="eval")
    return _eval_arithmetic(tree)


_MANUFACTURING_SECTORS = {
    "automobile", "metals & mining", "electricals", "capital goods", "cement",
}

_BANKING_SECTORS = {"banking"}


def _normalize_period_filter(period: Any = None, fiscal_year: Any = None) -> str | None:
    value = str(period if period is not None else (fiscal_year if fiscal_year is not None else "")).strip()
    if not value:
        return None
    if re.fullmatch(r"\d{4}", value):
        return "FY" + value
    return value


@lru_cache(maxsize=1)
def _sector_map() -> dict[str, str]:
    path = PROJECT_ROOT / "data" / "company_sectors.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, str] = {}
    for company, value in payload.items():
        if isinstance(value, dict):
            sector = value.get("sector")
        else:
            sector = value
        if sector is not None:
            out[str(company).strip()] = str(sector).strip()
    return out


@lru_cache(maxsize=1)
def _display_name_map() -> dict[str, str]:
    path = PROJECT_ROOT / "data" / "company_sectors.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, str] = {}
    for company, value in payload.items():
        if isinstance(value, dict):
            name = value.get("display_name") or company
        else:
            name = company
        out[str(company).strip()] = str(name).strip()
    return out


def display_name(company_id: str) -> str:
    cid = str(company_id or "").strip()
    if not cid:
        return ""
    mapped = _display_name_map().get(cid)
    if mapped:
        return str(mapped).strip()
    # fallback: preserve acronyms reasonably
    parts = cid.replace("_", " ").split()
    out = []
    for p in parts:
        if p.isupper() and len(p) <= 5:
            out.append(p)
        elif len(p) <= 4 and p.isalpha():
            out.append(p.upper())
        else:
            out.append(p.capitalize())
    return " ".join(out).strip()


@lru_cache(maxsize=1)
def _sector_label_map() -> dict[str, dict[str, list[str]]]:
    path = PROJECT_ROOT / "config" / "sector_label_map.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, list[str]]] = {}
    for sector, metrics in payload.items():
        if not isinstance(metrics, dict):
            continue
        norm_sector = str(sector).strip().lower()
        out[norm_sector] = {}
        for metric, labels in metrics.items():
            if isinstance(labels, list):
                out[norm_sector][str(metric).strip().lower()] = [str(l) for l in labels if str(l).strip()]
    return out


def _company_periods(path: str | None = None) -> dict[str, list[str]]:
    resolved = _resolved_index_path(path)
    rows = _meta_rows_resolved(resolved)
    out: dict[str, set[str]] = {}
    for row in rows:
        company = str(row[0]).strip()
        period = str(row[1]).strip()
        if company:
            out.setdefault(company, set()).add(period)
    return {company: sorted(values) for company, values in out.items()}


def list_companies(
    sector: str | None = None,
    fiscal_year: Any = None,
    period: str | None = None,
) -> dict[str, Any]:
    requested_period = _normalize_period_filter(period, fiscal_year)
    sector_filter = str(sector or "").strip()
    sectors = _sector_map()
    periods_by_company = _company_periods()
    companies: list[str] = []
    for company in company_ids():
        company_sector = sectors.get(company)
        if sector_filter:
            if company_sector is None:
                continue
            if sector_filter.lower() in {"manufacturing", "manufacturing sector"}:
                if company_sector.lower() not in _MANUFACTURING_SECTORS:
                    continue
            elif company_sector.lower() != sector_filter.lower():
                continue
        if requested_period:
            available = periods_by_company.get(company, [])
            if not any(
                str(item).strip().upper() == requested_period.upper()
                or str(item).strip().upper().startswith(requested_period.upper())
                for item in available
            ):
                continue
        companies.append(company)
    return {
        "companies": sorted(companies),
        "count": len(companies),
        "sector": sector_filter or None,
        "period": requested_period,
        "assumed_sectors": True,
    }


_METRIC_LABELS: dict[str, list[str]] = {
    "revenue": ["Revenue from operations", "Total revenue from operations", "Net sales", "Total revenue"],
    "revenue from operations": ["Revenue from operations", "Total revenue from operations", "Net sales", "Total revenue"],
    "total revenue": ["Revenue from operations", "Total revenue from operations", "Net sales", "Total revenue"],
    "total income": ["Total income"],
    "net sales": ["Revenue from operations", "Total revenue from operations", "Net sales", "Total revenue"],
    "operating expenses": ["Total expenses"],
    "total expenses": ["Total expenses"],
    "total costs": ["Total expenses"],
    "net profit": ["Net profit", "Profit after tax", "Profit for the year"],
    "profit after tax": ["Profit after tax", "Net profit", "Profit for the year"],
    "profit for the year": ["Profit for the year", "Profit after tax", "Net profit"],
    "total liabilities": ["Total liabilities"],
    "total equity and liabilities": ["Total equity and liabilities"],
    "non current liabilities": ["Total non-current liabilities", "Non-current liabilities"],
    "total non current liabilities": ["Total non-current liabilities", "Non-current liabilities"],
    "current liabilities": ["Total current liabilities", "Current liabilities"],
    "total current liabilities": ["Total current liabilities", "Current liabilities"],
}


_GET_METRIC_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_GET_METRIC_CACHE_LOCK = threading.Lock()


def clear_get_metric_cache() -> None:
    with _GET_METRIC_CACHE_LOCK:
        _GET_METRIC_CACHE.clear()


def _metric_cache_key(company: str, metric: str, period: str | None, reporting_basis: str | None, statement_kind: str | None = None) -> tuple[Any, ...]:
    try:
        from .extract import normalize_label as _normalize_label
        metric_key = _normalize_label(metric)
    except Exception:
        metric_key = str(metric or "").strip().lower()
    return (
        str(company or "").strip().lower(),
        metric_key,
        str(period or "").strip().upper() or None,
        str(reporting_basis or "").strip().lower() or None,
        str(statement_kind or "").strip().lower() or None,
    )


def get_metric(
    company: str,
    metric: str,
    fiscal_year: Any = None,
    period: str | None = None,
    statement_kind: str | None = None,
    reporting_basis: str | None = None,
) -> dict[str, Any]:
    from decimal import Decimal, InvalidOperation

    from .extract import _unit as _chunk_unit, extract_figures, normalize_label

    company = str(company or "").strip()
    requested_period = _normalize_period_filter(period, fiscal_year)
    normalized_metric = normalize_label(metric)
    cache_key = _metric_cache_key(company, metric, requested_period, reporting_basis, statement_kind)
    with _GET_METRIC_CACHE_LOCK:
        cached = _GET_METRIC_CACHE.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)
    budget.soft_check()

    sector = _sector_map().get(company, "")
    sector_labels = _sector_label_map().get(str(sector).lower(), {})
    banking_proxy = bool(sector and str(sector).lower() in _BANKING_SECTORS and normalized_metric in {
        "revenue", "revenue from operations", "total revenue", "total income", "net sales"
    })
    labels = sector_labels.get(normalized_metric) or _METRIC_LABELS.get(normalized_metric, [str(metric).strip()])
    # One filtered lookup per company/metric/period; prefer a canonical revenue query.
    canonical_query = "Revenue from operations" if "revenue" in normalized_metric else str(labels[0] if labels else metric).strip() or str(metric)
    periods_to_search: list[str | None] = [requested_period]
    if requested_period and re.fullmatch(r"FY\d{4}", requested_period.upper()):
        periods_to_search.append(f"FY{int(requested_period.upper()[2:])+1}")

    def _search_chunks(p_search: str | None) -> list[dict[str, Any]]:
        result = search_index(
            canonical_query,
            k=60,
            company=company,
            period=p_search,
            statement_kind=statement_kind,
            reporting_basis=reporting_basis,
            rerank_pool=12,
        )
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in result.get("hits", []) if isinstance(result, dict) else []:
            chunk_id = str(hit.get("chunk_id") or hit.get("id") or "").strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            out.append({
                "chunk_id": chunk_id,
                "text": str(hit.get("text") or hit.get("content") or ""),
                "meta": hit.get("meta") or {},
            })
        # add corresponding page chunks for table hits to recover headers
        for chunk in list(out):
            budget.soft_check()
            cid = str(chunk.get("chunk_id") or "")
            if ":t" not in cid:
                continue
            meta = chunk.get("meta") or {}
            page = meta.get("page")
            doc = str(meta.get("document_id") or cid.split(":")[0])
            company_id = str(meta.get("company_id") or company).strip().lower()
            if page is None or not doc or not company_id:
                continue
            try:
                page_cid = _page_chunk_map_resolved(str(_resolved_index_path())).get((company_id, doc, int(page)))
            except Exception:
                page_cid = None
            if not page_cid:
                page_cid = f"{doc}:p{int(page)}"
            if page_cid in seen:
                continue
            try:
                pc = get_chunk(page_cid)
            except Exception:
                continue
            seen.add(page_cid)
            out.append({
                "chunk_id": page_cid,
                "text": str(pc.get("text") or ""),
                "meta": pc.get("meta") or {},
            })
        return out

    def _resolve(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_rank: tuple[int, int, int, int] | None = None
        for label in labels:
            extracted = extract_figures(
                [{"company_id": company, "label": label, "period": requested_period}],
                chunks,
                basis=reporting_basis,
                statement_kind=statement_kind,
                period=requested_period,
            )
            for cand in extracted.get("figures") or []:
                c_basis = str(cand.get("reporting_basis") or "").lower()
                if reporting_basis is None and c_basis != "consolidated":
                    continue
                basis_rank = 0 if c_basis == "consolidated" else (1 if c_basis == "standalone" else 2)
                comp_rank = 1 if cand.get("comparative_column") else 0
                src_rank = 0 if str(cand.get("source_period") or "").upper() == str(requested_period or "").upper() else 1
                chunk_text = ""
                for ch in chunks:
                    if str(ch.get("chunk_id") or "") == str(cand.get("chunk_id") or ""):
                        chunk_text = str(ch.get("text") or "").lower()
                        break
                note_rank = 1 if ("notes to" in chunk_text[:180] or cand.get("note_chunk")) else 0
                rank = (basis_rank, src_rank, comp_rank, note_rank)
                if best is None or rank < best_rank:
                    best = dict(cand)
                    best_rank = rank
        if best is not None:
            return best
        if normalize_label(str(metric)) == "total assets":
            extracted = extract_figures(
                [{"company_id": company, "label": "Total equity and liabilities", "period": requested_period}],
                chunks,
                basis=reporting_basis,
                statement_kind=statement_kind,
                period=requested_period,
            )
            figures = extracted.get("figures") or []
            if figures:
                fig = dict(figures[0])
                fig["label_equivalence"] = True
                fig["label"] = str(metric)
                return fig
        return None

    def _statement_scan_chunks(p_search: str | None) -> list[dict[str, Any]]:
        """Scan matched company-period chunks in local index without embedding/rerank."""
        resolved_path = _resolved_index_path()
        index = _open_index(resolved_path)
        chunk_ids = json.loads((Path(resolved_path) / "chunk_ids.json").read_text(encoding="utf-8"))
        by_period = _chunk_ids_by_company_period(resolved_path)
        rows = _meta_rows_for_index(index, resolved_path)
        key = (company.lower(), str(p_search or "").upper())
        candidate_ids = set(by_period.get(key, ()))
        if not candidate_ids:
            for cid, row in zip(chunk_ids, rows, strict=False):
                if str(row[0]).strip().lower() == company.lower() and str(row[1]).strip().upper() == str(p_search or "").upper():
                    candidate_ids.add(str(cid))
        out: list[dict[str, Any]] = []
        seen_local: set[str] = set()
        wanted_label_norms = {normalize_label(l) for l in labels}
        for cid in candidate_ids:
            if cid in seen_local:
                continue
            seen_local.add(cid)
            try:
                chunk = get_chunk(cid)
            except Exception:
                continue
            meta = chunk.get("meta") or {}
            if str(meta.get("company_id") or "").strip().lower() != company.lower():
                continue
            text = str(chunk.get("text") or "")
            if not text:
                continue
            basis = str(meta.get("reporting_basis") or "").lower()
            sk = str(meta.get("statement_kind") or "").lower()
            text_low = text.lower()
            looks_pl = sk == "income_statement" or "statement of profit" in text_low or "profit and loss" in text_low or "income statement" in text_low
            if not looks_pl:
                continue
            if basis == "standalone" or sk in {"note", "segment"}:
                continue
            if "standalone" in text_low and "consolidated" not in text_low:
                continue
            if "consolidated" not in text_low and basis != "consolidated":
                continue
            head = " ".join(text_low.splitlines()[:20])
            if any(tok in head for tok in ("notes to", "note to", "notes forming part", "summarised", "summarized", "summary", "operating segment", "segment revenue", "reconciliation of revenue")):
                continue
            if text_low.lstrip().startswith("notes"):
                continue
            has_label = any(
                normalize_label(line.split(':',1)[0]) in wanted_label_norms
                or any(normalize_label(line.split(':',1)[0]) == normalize_label(lbl) for lbl in wanted_label_norms)
                for line in text.splitlines() if line.strip()
            )
            if not has_label:
                text_norm = normalize_label(text_low)
                has_label = any(normalize_label(lbl) in text_norm for lbl in wanted_label_norms)
            if not has_label and "revenue from operations" not in text_low and "total revenue" not in text_low:
                continue
            is_note = "notes to" in text_low[:180] or "note " in text_low[:60]
            out.append({"chunk_id": cid, "text": text, "meta": meta, "_is_note": is_note})
        out.sort(key=lambda c: (1 if c.get("_is_note") else 0, str(c.get("chunk_id") or "")))
        for c in out:
            c.pop("_is_note", None)
        return out

    chunks: list[dict[str, Any]] = []
    fig: dict[str, Any] | None = None
    searched_periods: list[str | None] = []
    for p_search in periods_to_search:
        budget.soft_check()
        searched_periods.append(p_search)
        new_chunks = _search_chunks(p_search)
        existing_ids = {str(c.get("chunk_id") or "") for c in chunks}
        for c in new_chunks:
            if str(c.get("chunk_id") or "") not in existing_ids:
                chunks.append(c)
        fig = _resolve(chunks)
        if fig is not None:
            break
    if fig is None and requested_period:
        for p_search in searched_periods:
            budget.soft_check()
            scan_chunks = _statement_scan_chunks(p_search)
            existing_ids = {str(c.get("chunk_id") or "") for c in chunks}
            for c in scan_chunks:
                if str(c.get("chunk_id") or "") not in existing_ids:
                    chunks.append(c)
        fig = _resolve(chunks)

    note = "banking revenue proxy" if banking_proxy else None
    if fig is not None and normalized_metric == "total liabilities" and "equity" in normalize_label(str(fig.get("label") or "")):
        fig = None
    if fig is not None:
        fig_doc = str((fig.get("chunk_id") or "")).split(":")[0]
        fig_period = str(fig.get("period") or requested_period or "").strip().lower()
        if str(fig.get("unit") or "").strip() in {"", "unit not stated in source"}:
            for chunk in chunks:
                chunk_meta = chunk.get("meta") or {}
                if str(chunk_meta.get("company_id") or "").strip() != str(fig.get("company_id") or company).strip():
                    continue
                c_doc = str(chunk_meta.get("document_id") or (str(chunk.get("chunk_id") or "").split(":")[0])).strip()
                c_period = str(chunk_meta.get("period") or "").strip().lower()
                src_period = str(fig.get("source_period") or "").strip().lower()
                allowed_periods = {p for p in (fig_period, src_period) if p}
                if allowed_periods and c_period and c_period not in allowed_periods:
                    continue
                if fig_doc and c_doc and c_doc != fig_doc:
                    continue
                c_page = chunk_meta.get("page")
                fig_page = fig.get("page")
                if fig_page is not None and c_page is not None:
                    try:
                        if abs(int(c_page) - int(fig_page)) > 1:
                            continue
                    except (TypeError, ValueError):
                        if str(c_page) != str(fig_page):
                            continue
                candidate_unit = _chunk_unit(chunk)
                if candidate_unit and candidate_unit != "unit not stated in source":
                    fig["unit"] = candidate_unit
                    fig["unit_source"] = "same_page_header"
                    break
        if fig.get("comparative_column"):
            note = ((note + "; ") if note else "") + "comparative column"
        printed_value = fig.get("value")
        printed_unit = fig.get("unit")
        label_used = str(fig.get("label") or metric)
        value = fig.get("value")
        unit = fig.get("unit")
        normalized = False
        is_margin = "margin" in normalized_metric
        is_growth = "growth" in normalized_metric
        if is_margin or is_growth:
            try:
                value = round(float(str(value).replace(",", "")), 2)
            except Exception:
                pass
            unit = "percent"
            printed_unit = printed_unit or unit
        else:
            if str(unit or "").strip() in {"", "unit not stated in source"}:
                unit = "unit not stated in source"
                note = ((note + "; ") if note else "") + "unit not stated in source"
            else:
                if fig.get("unit_source") == "same_page_header":
                    note = ((note + "; ") if note else "") + "unit inherited from page header"
                value, unit, normalized = _normalize_value_to_crore(value, unit)
        doc_id = str(fig.get("document_id") or fig_doc or "").strip() or None
        page = fig.get("page")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        result = {
            "company_id": company,
            "display_name": display_name(company),
            "metric": metric,
            "label_used": label_used,
            "printed_value": printed_value,
            "printed_unit": printed_unit,
            "value": value,
            "unit": unit,
            "citation": fig.get("chunk_id"),
            "document_id": doc_id,
            "page": page,
            "period": fig.get("period") or requested_period,
            "source_period": fig.get("source_period"),
            "column_header": fig.get("column_header") or fig.get("statement_header"),
            "comparative_column": bool(fig.get("comparative_column")),
            "reporting_basis": fig.get("reporting_basis") or reporting_basis,
            "statement_kind": fig.get("statement_kind"),
            "missing": False,
            "derived": False,
            "note": note,
            "sector": sector,
            "normalized_to_crore": normalized,
            "excluded_for_ranking": False,
            "statement_header": fig.get("statement_header"),
            "raw_row": fig.get("raw_row"),
        }
        with _GET_METRIC_CACHE_LOCK:
            _GET_METRIC_CACHE[cache_key] = copy.deepcopy(result)
        return result
    if normalized_metric == "total liabilities":
        def _is_equity_inclusive(label: str) -> bool:
            low = normalize_label(label)
            return "equity" in low and "liabilit" in low

        for basis_try in ([reporting_basis] if reporting_basis else ["consolidated", "standalone"]):
            nc = get_metric(company, "Total non-current liabilities", fiscal_year=fiscal_year, period=period, statement_kind=statement_kind, reporting_basis=basis_try)
            cu = get_metric(company, "Total current liabilities", fiscal_year=fiscal_year, period=period, statement_kind=statement_kind, reporting_basis=basis_try)
            if nc.get("missing") or cu.get("missing"):
                continue
            if str(nc.get("period","")).upper() != str(cu.get("period","")).upper():
                continue
            nc_unit = str(nc.get("unit") or "")
            cu_unit = str(cu.get("unit") or "")
            if nc_unit != cu_unit or "not stated" in nc_unit.lower():
                continue
            try:
                total_val = Decimal(str(nc.get("value")).replace(",", "")) + Decimal(str(cu.get("value")).replace(",", ""))
            except Exception:
                continue
            out_val = format(total_val.normalize(), "f")
            if "." in out_val:
                out_val = out_val.rstrip("0").rstrip(".")
            result = {
                "company_id": company,
                "display_name": display_name(company),
                "metric": metric,
                "label_used": "Computed: Total non-current liabilities + Total current liabilities",
                "printed_value": out_val,
                "printed_unit": nc_unit,
                "value": out_val,
                "unit": nc_unit,
                "citation": nc.get("citation"),
                "document_id": nc.get("document_id"),
                "page": nc.get("page"),
                "period": nc.get("period") or requested_period,
                "source_period": nc.get("source_period"),
                "column_header": nc.get("column_header") or nc.get("statement_header"),
                "comparative_column": bool(nc.get("comparative_column") or cu.get("comparative_column")),
                "reporting_basis": nc.get("reporting_basis") or basis_try,
                "statement_kind": nc.get("statement_kind"),
                "missing": False,
                "derived": True,
                "note": "Computed from printed balance-sheet components; not a separately printed total-liabilities line.",
                "sector": sector,
                "normalized_to_crore": bool(nc.get("normalized_to_crore") and cu.get("normalized_to_crore")),
                "excluded_for_ranking": False,
                "statement_header": nc.get("statement_header"),
                "raw_row": None,
                "components": {
                    "non_current_liabilities": {"value": nc.get("value"), "unit": nc_unit, "citation": nc.get("citation"), "page": nc.get("page"), "document_id": nc.get("document_id"), "label_used": nc.get("label_used")},
                    "current_liabilities": {"value": cu.get("value"), "unit": cu_unit, "citation": cu.get("citation"), "page": cu.get("page"), "document_id": cu.get("document_id"), "label_used": cu.get("label_used")},
                },
            }
            with _GET_METRIC_CACHE_LOCK:
                _GET_METRIC_CACHE[cache_key] = copy.deepcopy(result)
            return result

    if normalized_metric in {"operating profit", "operating profit margin"}:
        margin_requested = normalized_metric == "operating profit margin"
        # enforce same basis: try consolidated then standalone
        chosen_basis = reporting_basis
        revenue = expenses = finance = None
        for basis_try in ([reporting_basis] if reporting_basis else ["consolidated", "standalone"]):
            r = get_metric(company, "Revenue from operations", fiscal_year=fiscal_year, period=period, statement_kind=statement_kind, reporting_basis=basis_try)
            e = get_metric(company, "Total expenses", fiscal_year=fiscal_year, period=period, statement_kind=statement_kind, reporting_basis=basis_try)
            f = get_metric(company, "Finance cost", fiscal_year=fiscal_year, period=period, statement_kind=statement_kind, reporting_basis=basis_try)
            if not r.get("missing") and not e.get("missing") and not f.get("missing"):
                # same period/doc basis already ensured by same basis_try; ensure same period
                if str(r.get("period","")).upper()==str(e.get("period","")).upper()==str(f.get("period","")).upper():
                    revenue, expenses, finance, chosen_basis = r, e, f, basis_try
                    break
        if revenue is None:
            # fallback individual to report missing
            revenue = get_metric(company, "Revenue from operations", fiscal_year=fiscal_year, period=period, statement_kind=statement_kind, reporting_basis=reporting_basis)
            expenses = get_metric(company, "Total expenses", fiscal_year=fiscal_year, period=period, statement_kind=statement_kind, reporting_basis=reporting_basis)
            finance = get_metric(company, "Finance cost", fiscal_year=fiscal_year, period=period, statement_kind=statement_kind, reporting_basis=reporting_basis)
        if margin_requested:
            rev_missing = revenue.get("missing")
            exp_label = str(expenses.get("label_used") or "").lower()
            exp_missing = expenses.get("missing") or not normalize_label(exp_label).startswith("total expenses")
            fin_missing = finance.get("missing")
            if rev_missing or exp_missing or fin_missing:
                missing_result = {
                    "company_id": company,
                    "display_name": display_name(company),
                    "metric": metric,
                    "label_used": "Derived: (Revenue - Total expenses + Finance costs) / Revenue",
                    "value": None,
                    "unit": "percent",
                    "citation": [c for c in (revenue.get("citation"), expenses.get("citation"), finance.get("citation")) if c],
                    "document_id": revenue.get("document_id") or expenses.get("document_id") or finance.get("document_id"),
                    "page": revenue.get("page") or expenses.get("page") or finance.get("page"),
                    "period": requested_period,
                    "reporting_basis": revenue.get("reporting_basis"),
                    "statement_kind": revenue.get("statement_kind"),
                    "missing": True,
                    "derived": True,
                    "formula": "(Revenue - Total expenses + Finance costs) / Revenue * 100",
                    "note": (("standalone; " if str(chosen_basis).lower()=="standalone" else "") + ("not available: missing margin inputs" if (rev_missing or fin_missing or expenses.get("missing")) else "not available: expense label not full total")).strip(),
                    "sector": sector,
                    "components": {
                        "revenue": {"value": revenue.get("value"), "unit": revenue.get("unit"), "missing": rev_missing},
                        "total_expenses": {"value": expenses.get("value"), "unit": expenses.get("unit"), "missing": exp_missing, "label_used": expenses.get("label_used")},
                        "finance_costs": {"value": finance.get("value"), "unit": finance.get("unit"), "missing": fin_missing},
                    },
                }
                with _GET_METRIC_CACHE_LOCK:
                    _GET_METRIC_CACHE[cache_key] = copy.deepcopy(missing_result)
                return missing_result
            try:
                rev = Decimal(str(revenue.get("value")).replace(",", ""))
                exp = Decimal(str(expenses.get("value")).replace(",", ""))
                fin = Decimal(str(finance.get("value")).replace(",", ""))
                margin = (rev - exp + fin) / rev * 100 if rev != 0 else None
                margin_val = round(float(margin), 2) if margin is not None else None
                out_of_range = margin_val is not None and (margin_val < -50 or margin_val > 60)
                result = {
                    "company_id": company,
                    "metric": metric,
                    "label_used": "Derived: (Revenue - Total expenses + Finance costs) / Revenue",
                    "value": margin_val,
                    "unit": "percent",
                    "citation": [c for c in (revenue.get("citation"), expenses.get("citation"), finance.get("citation")) if c],
                    "document_id": revenue.get("document_id") or expenses.get("document_id") or finance.get("document_id"),
                    "page": revenue.get("page") or expenses.get("page") or finance.get("page"),
                    "period": requested_period,
                    "reporting_basis": revenue.get("reporting_basis"),
                    "statement_kind": revenue.get("statement_kind"),
                    "missing": bool(out_of_range or margin_val is None),
                    "derived": True,
                    "formula": "(Revenue - Total expenses + Finance costs) / Revenue * 100",
                    "note": (("standalone; " if str(chosen_basis).lower()=="standalone" else "") + ("check extraction" if out_of_range else "")).strip("; ") or None,
                    "sector": sector,
                    "components": {
                        "revenue": {"value": revenue.get("value"), "unit": revenue.get("unit"), "missing": False, "label_used": revenue.get("label_used")},
                        "total_expenses": {"value": expenses.get("value"), "unit": expenses.get("unit"), "missing": False, "label_used": expenses.get("label_used"), "citation": expenses.get("citation")},
                        "finance_costs": {"value": finance.get("value"), "unit": finance.get("unit"), "missing": False, "label_used": finance.get("label_used")},
                    },
                }
                with _GET_METRIC_CACHE_LOCK:
                    _GET_METRIC_CACHE[cache_key] = copy.deepcopy(result)
                return result
            except Exception:
                missing_result = {
                    "company_id": company,
                    "display_name": display_name(company),
                    "metric": metric,
                    "label_used": "Derived: (Revenue - Total expenses + Finance costs) / Revenue",
                    "value": None,
                    "unit": "percent",
                    "citation": [c for c in (revenue.get("citation"), expenses.get("citation"), finance.get("citation")) if c],
                    "document_id": revenue.get("document_id") or expenses.get("document_id") or finance.get("document_id"),
                    "page": revenue.get("page") or expenses.get("page") or finance.get("page"),
                    "period": requested_period,
                    "reporting_basis": revenue.get("reporting_basis"),
                    "statement_kind": revenue.get("statement_kind"),
                    "missing": True,
                    "derived": True,
                    "formula": "(Revenue - Total expenses + Finance costs) / Revenue * 100",
                    "note": (("standalone; " if str(chosen_basis).lower()=="standalone" else "") + "calculation failed").strip("; "),
                    "sector": sector,
                    "components": {
                        "revenue": {"value": revenue.get("value"), "unit": revenue.get("unit"), "missing": revenue.get("missing")},
                        "total_expenses": {"value": expenses.get("value"), "unit": expenses.get("unit"), "missing": expenses.get("missing"), "label_used": expenses.get("label_used")},
                        "finance_costs": {"value": finance.get("value"), "unit": finance.get("unit"), "missing": finance.get("missing")},
                    },
                }
                with _GET_METRIC_CACHE_LOCK:
                    _GET_METRIC_CACHE[cache_key] = copy.deepcopy(missing_result)
                return missing_result
            except (InvalidOperation, ValueError):
                pass
        if not expenses.get("missing") and not revenue.get("missing"):
            try:
                rev = Decimal(str(revenue.get("value")).replace(",", ""))
                exp = Decimal(str(expenses.get("value")).replace(",", ""))
                fin = Decimal("0")
                if not finance.get("missing"):
                    fin = Decimal(str(finance.get("value")).replace(",", ""))
                operating = rev - exp + fin
                citations = [c for c in (revenue.get("citation"), expenses.get("citation"), finance.get("citation")) if c]
                op_value, op_unit, op_norm = _normalize_value_to_crore(str(operating), revenue.get("unit"))
                return {
                    "company_id": company,
                    "metric": metric,
                    "label_used": "Derived: Revenue from operations - Total expenses + Finance cost",
                    "value": op_value,
                    "unit": op_unit,
                    "citation": citations,
                    "document_id": revenue.get("document_id") or expenses.get("document_id") or finance.get("document_id"),
                    "page": revenue.get("page") or expenses.get("page") or finance.get("page"),
                    "period": requested_period,
                    "reporting_basis": revenue.get("reporting_basis"),
                    "statement_kind": revenue.get("statement_kind"),
                    "missing": False,
                    "derived": True,
                    "formula": "Revenue from operations - Total expenses + Finance cost",
                    "note": "banking revenue proxy" if banking_proxy else None,
                    "sector": sector,
                    "normalized_to_crore": op_norm,
                }
            except (InvalidOperation, ValueError):
                pass
    missing_result = {
        "company_id": company,
        "display_name": display_name(company),
        "metric": metric,
        "label_used": str(metric),
        "value": None,
        "unit": None,
        "citation": None,
        "period": requested_period,
        "missing": True,
        "derived": False,
        "note": note,
        "sector": sector,
    }
    with _GET_METRIC_CACHE_LOCK:
        _GET_METRIC_CACHE[cache_key] = copy.deepcopy(missing_result)
    return missing_result


def compute_growth(current: Any, prior: Any) -> dict[str, Any]:
    from decimal import Decimal, InvalidOperation

    try:
        cur = Decimal(str(current).replace(",", ""))
        prev = Decimal(str(prior).replace(",", ""))
        if prev == 0:
            return {"growth_percent": None, "error": "prior is zero"}
        return {"growth_percent": round(float((cur - prev) / prev * 100), 2)}
    except (InvalidOperation, ValueError):
        return {"growth_percent": None, "error": "invalid number"}


def compute_margin(numerator: Any, denominator: Any) -> dict[str, Any]:
    from decimal import Decimal, InvalidOperation

    try:
        num = Decimal(str(numerator).replace(",", ""))
        den = Decimal(str(denominator).replace(",", ""))
        if den == 0:
            return {"margin_percent": None, "error": "denominator is zero"}
        return {"margin_percent": round(float(num / den * 100), 2)}
    except (InvalidOperation, ValueError):
        return {"margin_percent": None, "error": "invalid number"}


def rank_values(items: Any) -> dict[str, Any]:
    if not isinstance(items, list):
        return {"ranked": [], "error": "items must be a list"}
    ranked = sorted(
        [item for item in items if isinstance(item, dict)],
        key=lambda item: _to_decimal_safe(item.get("value")),
        reverse=True,
    )
    return {"ranked": ranked, "count": len(ranked)}



def rank_companies(
    metric: str,
    fiscal_year: Any = None,
    period: str | None = None,
    sector: str | None = None,
    n: int = 10,
    statement_kind: str | None = None,
    reporting_basis: str | None = None,
    on_company: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from decimal import Decimal

    requested_period = _normalize_period_filter(period, fiscal_year)
    companies_info = list_companies(sector=sector, fiscal_year=fiscal_year, period=requested_period)
    companies = companies_info.get("companies", [])
    # dedupe by canonical company id
    seen_companies=set()
    deduped=[]
    for c in companies:
        canon=str(c).strip()
        key=canon.lower()
        if key in seen_companies:
            continue
        seen_companies.add(key)
        deduped.append(canon)
    companies=deduped
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for company in companies:
        t0 = time.monotonic()
        try:
            budget.soft_check()
        except Exception as exc:
            reason = f"Stopped by soft deadline: {exc}"
            missing.append({"company_id": company, "reason": reason})
            if on_company is not None:
                on_company({"company_id": company, "display_name": display_name(company), "missing": True, "reason": reason, "latency_ms": int((time.monotonic()-t0)*1000)})
            continue
        result = get_metric(
            company, metric, fiscal_year=fiscal_year, period=requested_period,
            statement_kind=statement_kind, reporting_basis=reporting_basis,
        )
        if result.get("missing"):
            reason = str(result.get("reason") or "No consolidated statement figure with matching year column header")
            missing.append({"company_id": company, "reason": reason})
            if on_company is not None:
                on_company({"company_id": company, "display_name": display_name(company), "missing": True, "reason": reason, "period": result.get("period"), "citation": result.get("citation"), "latency_ms": int((time.monotonic()-t0)*1000)})
            continue
        actual_period = str(result.get("period") or "").strip().upper()
        if actual_period and requested_period and actual_period != str(requested_period).upper():
            reason = f"Period mismatch: got {actual_period}"
            missing.append({"company_id": company, "reason": reason})
            if on_company is not None:
                on_company({"company_id": company, "display_name": display_name(company), "missing": True, "reason": reason, "period": result.get("period"), "citation": result.get("citation"), "latency_ms": int((time.monotonic()-t0)*1000)})
            continue
        unit = str(result.get("unit") or "")
        lower_unit = unit.lower()
        try:
            raw_value = Decimal(str(result.get("value")).replace(",", ""))
        except Exception:
            raw_value = None
        converted = None
        is_margin = "margin" in str(metric).lower() or lower_unit == "percent"
        if raw_value is not None and is_margin:
            converted = raw_value
        elif raw_value is not None:
            if result.get("normalized_to_crore") or lower_unit == "rs. crore":
                converted = raw_value
            elif "million" in lower_unit or lower_unit.endswith("mn") or " mn" in lower_unit:
                converted = raw_value / Decimal("10")
            elif "lakh" in lower_unit or "lac" in lower_unit:
                converted = raw_value / Decimal("100")
            elif "crore" in lower_unit or " cr" in lower_unit or lower_unit.endswith("cr"):
                converted = raw_value
        if converted is None or result.get("excluded_for_ranking"):
            reason = str(result.get("reason") or "Unknown or unverified unit; excluded from ranking")
            missing.append({"company_id": company, "reason": reason})
            if on_company is not None:
                on_company({"company_id": company, "display_name": display_name(company), "missing": True, "reason": reason, "period": result.get("period"), "citation": result.get("citation"), "latency_ms": int((time.monotonic()-t0)*1000)})
            continue
        row = {
            "company_id": str(result.get("company_id") or company).strip(),
            "display_name": str(result.get("display_name") or display_name(company)),
            "value": result.get("value"),
            "printed_value": result.get("printed_value"),
            "printed_unit": result.get("printed_unit"),
            "unit": unit,
            "value_in_crore": str(converted),
            "citation": result.get("citation"),
            "document_id": result.get("document_id"),
            "page": result.get("page"),
            "period": result.get("period"),
            "source_period": result.get("source_period"),
            "reporting_basis": result.get("reporting_basis"),
            "derived": result.get("derived", False),
            "label_used": result.get("label_used"),
            "note": result.get("note"),
            "sector": result.get("sector") or _sector_map().get(company),
            "column_header": result.get("column_header"),
            "comparative_column": bool(result.get("comparative_column")),
            "raw_row": result.get("raw_row"),
        }
        if isinstance(result.get("components"), dict):
            row["components"] = result["components"]
        if "revenue" in str(metric).lower() and not str(row.get("sector") or "").strip():
            reason = "Sector metadata unavailable"
            missing.append({"company_id": company, "reason": reason})
            if on_company is not None:
                on_company({"company_id": company, "display_name": display_name(company), "missing": True, "reason": reason, "latency_ms": int((time.monotonic()-t0)*1000)})
            continue
        rows.append(row)
        if on_company is not None:
            on_company({
                "company_id": row["company_id"],
                "display_name": row["display_name"],
                "missing": False,
                "value": row.get("value"),
                "unit": row.get("unit"),
                "citation": row.get("citation"),
                "period": row.get("period"),
                "latency_ms": int((time.monotonic()-t0)*1000),
            })
    def _key(item: dict[str, Any]) -> Decimal:
        for field in ("value_in_crore", "value"):
            try:
                return Decimal(str(item.get(field)).replace(",", ""))
            except Exception:
                continue
        return Decimal("-Infinity")
    rows.sort(key=_key, reverse=True)
    limit = max(1, int(n or 10))
    return {
        "metric": metric,
        "period": requested_period,
        "sector": sector,
        "ranked": rows[:limit],
        "count_covered": len(companies),
        "count_found": len(rows),
        "missing": missing,
        "truncated": len(rows) > limit,
    }


def sum_values(items: Any) -> dict[str, Any]:
    from decimal import Decimal, InvalidOperation

    total = Decimal("0")
    count = 0
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("check_extraction"):
            continue
        try:
            total += Decimal(str(item.get("value") if isinstance(item, dict) else item).replace(",", ""))
            count += 1
        except (InvalidOperation, ValueError):
            continue
    return {"total": str(total), "count": count}

def detect_sector(text: str) -> str | None:
    lower = str(text or "").lower()
    aliases = {
        "healthcare": "Pharmaceuticals",
        "pharmaceutical": "Pharmaceuticals",
        "pharma": "Pharmaceuticals",
        "automobile": "Automobile",
        "auto": "Automobile",
        "energy": "Energy",
        "banking": "Banking",
        "manufacturing": "Manufacturing",
        "chemicals": "Chemicals",
        "chemical": "Chemicals",
        "fmcg": "FMCG",
        "it": "IT",
        "telecom": "Telecom",
        "insurance": "Insurance",
    }
    for alias, sector in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(r"\b" + re.escape(alias) + r"\b", lower):
            return sector
    for sector in sorted(set(_sector_map().values()), key=len, reverse=True):
        if re.search(r"\b" + re.escape(sector.lower()) + r"\b", lower):
            return sector
    return None



def compare_periods(
    metric: str,
    base_period: str,
    compare_period: str,
    sector: str | None = None,
    fiscal_year: Any = None,
    statement_kind: str | None = None,
    reporting_basis: str | None = None,
    on_company: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    base_p = _normalize_period_filter(base_period, fiscal_year)
    compare_p = _normalize_period_filter(compare_period, fiscal_year)
    companies_info = list_companies(sector=sector, period=compare_p)
    # include companies present in either period
    base_companies = set(list_companies(sector=sector, period=base_p).get("companies", []))
    compare_companies = set(companies_info.get("companies", []))
    companies = sorted(base_companies | compare_companies)
    rows: list[dict[str, Any]] = []
    for company in companies:
        t0 = time.monotonic()
        try:
            budget.soft_check()
        except Exception as exc:
            dn = display_name(company)
            row = {
                "company_id": company, "display_name": dn, "base_period": base_p, "compare_period": compare_p,
                "base_value": None, "compare_value": None,
                "growth_percent": None, "base_citation": None, "compare_citation": None,
                "base_document_id": None, "compare_document_id": None,
                "base_page": None, "compare_page": None,
                "base_period_actual": None, "compare_period_actual": None,
                "unit": None, "missing": True,
                "reporting_basis": reporting_basis,
                "note": f"Stopped by soft deadline: {exc}", "check_extraction": False,
            }
            rows.append(row)
            if on_company is not None:
                on_company({"company_id": company, "display_name": dn, "missing": True, "reason": row["note"], "latency_ms": int((time.monotonic()-t0)*1000)})
            continue
        base_result = get_metric(
            company, metric, period=base_p, statement_kind=statement_kind,
            reporting_basis=reporting_basis,
        )
        compare_result = get_metric(
            company, metric, period=compare_p, statement_kind=statement_kind,
            reporting_basis=reporting_basis,
        )
        base_period_actual = str(base_result.get("period") or "").strip().upper()
        compare_period_actual = str(compare_result.get("period") or "").strip().upper()
        if (base_result.get("missing") or compare_result.get("missing")
                or base_period_actual not in {"", str(base_p).upper()}
                or compare_period_actual not in {"", str(compare_p).upper()}):
            # normalize FY prefix
            if base_period_actual and base_period_actual != str(base_p).upper():
                base_result["missing"] = True
            if compare_period_actual and compare_period_actual != str(compare_p).upper():
                compare_result["missing"] = True
        dn = display_name(company)
        if base_result.get("missing") or compare_result.get("missing"):
            row = {
                "company_id": company, "display_name": dn, "base_period": base_p, "compare_period": compare_p,
                "base_value": base_result.get("value"), "compare_value": compare_result.get("value"),
                "growth_percent": None, "base_citation": base_result.get("citation"), "compare_citation": compare_result.get("citation"),
                "base_document_id": base_result.get("document_id"), "compare_document_id": compare_result.get("document_id"),
                "base_page": base_result.get("page"), "compare_page": compare_result.get("page"),
                "base_period_actual": base_result.get("period"), "compare_period_actual": compare_result.get("period"),
                "unit": base_result.get("unit") or compare_result.get("unit"), "missing": True,
                "reporting_basis": base_result.get("reporting_basis") or compare_result.get("reporting_basis"),
                "note": base_result.get("note") or compare_result.get("note"),
            }
            rows.append(row)
            if on_company is not None:
                on_company({"company_id": company, "display_name": dn, "missing": True, "reason": row.get("note"), "base_citation": row.get("base_citation"), "compare_citation": row.get("compare_citation"), "latency_ms": int((time.monotonic()-t0)*1000)})
            continue
        growth = compute_growth(compare_result.get("value"), base_result.get("value"))
        growth_pct = growth.get("growth_percent")
        check_extraction = isinstance(growth_pct, (int, float)) and (growth_pct > 100 or growth_pct < -50)
        note = base_result.get("note") or compare_result.get("note")
        if check_extraction:
            note = ((str(note) + "; ") if note else "") + "check extraction"
        row = {
            "company_id": company, "display_name": dn, "base_period": base_p, "compare_period": compare_p,
            "base_value": base_result.get("value"), "compare_value": compare_result.get("value"),
            "growth_percent": growth_pct, "base_citation": base_result.get("citation"), "compare_citation": compare_result.get("citation"),
            "base_document_id": base_result.get("document_id"), "compare_document_id": compare_result.get("document_id"),
            "base_page": base_result.get("page"), "compare_page": compare_result.get("page"),
            "base_period_actual": base_result.get("period"), "compare_period_actual": compare_result.get("period"),
            "unit": base_result.get("unit") or compare_result.get("unit"), "missing": False,
            "reporting_basis": base_result.get("reporting_basis") or compare_result.get("reporting_basis"),
            "note": note, "check_extraction": check_extraction,
        }
        rows.append(row)
        if on_company is not None:
            on_company({"company_id": company, "display_name": dn, "missing": False, "growth_percent": growth_pct, "base_citation": row.get("base_citation"), "compare_citation": row.get("compare_citation"), "latency_ms": int((time.monotonic()-t0)*1000)})
    return {
        "metric": metric,
        "base_period": base_p,
        "compare_period": compare_p,
        "sector": sector,
        "rows": rows,
        "count_covered": len(companies),
        "count_found": sum(1 for row in rows if not row.get("missing")),
    }



def _normalize_value_to_crore(value: Any, unit: Any) -> tuple[Any, str | None, bool]:
    from decimal import Decimal, InvalidOperation
    raw_unit = str(unit or "").strip()
    lower = raw_unit.lower()
    try:
        dec = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return value, raw_unit or None, False
    converted = None
    if any(t in lower for t in ["million", "millions", "mn"]):
        converted = dec / Decimal("10")
    elif any(t in lower for t in ["lakh", "lakhs", "lac", "lacs"]):
        converted = dec / Decimal("100")
    elif "crore" in lower or re.search(r"\bcr\.?\b", lower):
        converted = dec
    elif raw_unit in {"", "unit not stated in source"}:
        # do not guess here; infer_unit_from_document may handle
        return value, raw_unit or None, False
    if converted is None:
        return value, raw_unit or None, False
    out = format(converted.normalize(), "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return out, "Rs. crore", True


def _to_decimal_safe(value: Any) -> Any:
    from decimal import Decimal, InvalidOperation

    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return Decimal("-Infinity")
