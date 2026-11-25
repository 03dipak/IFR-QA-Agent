from __future__ import annotations

import ast
import json
import logging
import operator
import os
import re
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_INDEX_PATH = "results/full_index_v1/page_hash384"
LOGGER = logging.getLogger(__name__)


def _index_path() -> Path:
    return Path(os.environ.get("AGENT_INDEX_PATH", DEFAULT_INDEX_PATH))


def _resolved_index_path(path: str | None = None) -> str:
    return str((Path(path) if path else _index_path()).resolve())


@lru_cache(maxsize=4)
def _open_index_resolved(resolved_path: str) -> Any:
    from src.rag_ingestion.full_index import FullVectorIndex

    return FullVectorIndex(Path(resolved_path))


def _open_index(path: str | None = None) -> Any:
    return _open_index_resolved(_resolved_index_path(path))


def warm_index() -> None:
    path = _resolved_index_path()
    index = _open_index(path)
    _meta_rows_for_index(index, path)


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
        if isinstance(values, (list, tuple)):
            return tuple(sorted(
                value.strip() for value in values
                if isinstance(value, str) and value.strip()
            ))
    ids = _ids_from_meta(_index_meta(resolved_path))
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


@lru_cache(maxsize=4)
def _meta_rows_resolved(resolved_path: str) -> tuple[tuple[str, str, str, str, str, str], ...]:
    root = Path(resolved_path)
    ids = json.loads((root / "chunk_ids.json").read_text(encoding="utf-8"))
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
    return tuple(rows.get(str(chunk_id), empty) for chunk_id in ids)


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
    vector = embed_text(query, dim, embedder)
    return np.asarray(vector, dtype=np.float32)


_GENERIC_COMPANY_TOKENS = {"green", "pharma", "steel", "sun"}
_COMPANY_STOPWORDS = {
    "about", "after", "against", "and", "annual", "balance", "between", "cash",
    "compare", "compared", "company", "companies", "consolidated", "crore",
    "difference", "ended", "equity", "figure", "figures", "financial", "flow",
    "for", "from", "higher", "how", "income", "larger", "many", "march",
    "million", "much", "operations", "period", "printed", "profit", "quarter",
    "revenue", "standalone", "statement", "statements", "the", "their", "total",
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


def search_index(
    query: str,
    k: int = 5,
    company: str | None = None,
    period: str | None = None,
    statement_kind: str | None = None,
    doc_type: str | None = None,
    reporting_basis: str | None = None,
) -> dict[str, Any]:
    global _SEARCH_CALL_COUNT
    _SEARCH_CALL_COUNT += 1
    call_number = _SEARCH_CALL_COUNT
    started = time.monotonic()
    filtered = bool(company or period or statement_kind or doc_type or reporting_basis)
    try:
        if k <= 0:
            return {"hits": [], "match_info": {"excluded_empty_period": 0}}
        requested_basis = _normalize_basis(reporting_basis)
        path = _resolved_index_path()
        index = _open_index(path)
        has_filters = bool(
            company or period or statement_kind or doc_type or requested_basis
        )
        query_vector = _query_embedding(query, index, path) if has_filters else None
        relaxed: list[str] = []
        cur_period, cur_sk, cur_dt, cur_basis = (
            period, statement_kind, doc_type, requested_basis
        )
        hits, match_info = _search_once(
            query, k, company, cur_period, cur_sk, cur_dt, cur_basis, index, query_vector
        )
        if not hits and cur_sk:
            relaxed.append("drop_statement_kind")
            cur_sk = None
            hits, info = _search_once(
                query, k, company, cur_period, cur_sk, cur_dt, cur_basis, index, query_vector
            )
            _merge_info(match_info, info)
        if not hits and cur_basis:
            relaxed.append("drop_reporting_basis")
            cur_basis = None
            hits, info = _search_once(
                query, k, company, cur_period, cur_sk, cur_dt, cur_basis, index, query_vector
            )
            _merge_info(match_info, info)
        if not hits and cur_period:
            relaxed.append("drop_period")
            cur_period = None
            hits, info = _search_once(
                query, k, company, cur_period, cur_sk, cur_dt, cur_basis, index, query_vector
            )
            _merge_info(match_info, info)
        match_info["relaxed"] = relaxed
        match_info["final_period_filter"] = cur_period
        match_info["final_statement_kind_filter"] = cur_sk
        match_info["final_doc_type_filter"] = cur_dt
        match_info["final_reporting_basis_filter"] = cur_basis
        return {"hits": hits, "match_info": match_info}
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


def get_chunk(chunk_id: str) -> dict[str, Any]:
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
