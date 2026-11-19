"""Context builders for B1 (oracle) and B2 (global/filtered RAG)."""

from __future__ import annotations

import json
from pathlib import Path

from src.rag_ingestion.index import BM25Index
from src.eval.retrieval_sanity import page_usable, _filter_chunks
from src.resolver.query_resolver import resolve_query

PAGE_DIR = Path("data/corpus/extract/page_extracts")

# Endpoint context ceiling is 8192 tokens (prompt+completion). Pinned max_tokens=512,
# so prompts are capped here: 12000 chars is roughly 3000 tokens of context.
CONTEXT_CHAR_CAP = 12_000


def _page_text(doc_id: str, page: int) -> str:
    path = PAGE_DIR / f"{doc_id}.pages.jsonl"
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("pdf_page_index") == page:
            return rec.get("page_text") or ""
    return ""


def oracle_context(row: dict) -> str:
    """B1: text of the row's gold evidence pages (smoke only, not scored)."""
    blocks = []
    for e in row.get("evidence") or []:
        did = e.get("document_id")
        pg = e.get("pdf_page")
        if not did or not isinstance(pg, int):
            continue
        text = _page_text(did, pg)
        if text:
            blocks.append(f"[document_id={did} page={pg} usable={page_usable(did, pg)}]\n{text}")
    return "\n\n".join(blocks)


_IDX_CACHE: dict[tuple[int, bool], BM25Index] = {}


def rag_context(row: dict, chunks: list[dict], *, filtered: bool = False,
                top_k: int = 5) -> str:
    """B2: BM25 context. filtered=True restricts by resolved slots first."""
    if filtered:
        res = resolve_query(row.get("question_text") or row.get("question") or "")
        # build a temporary row-shaped filter from resolved slots
        frow = dict(row)
        if res.company_ids:
            frow["company_id"] = "|".join(res.company_ids)
        if res.periods:
            frow["fiscal_year"] = res.periods[0].token
        if res.basis:
            frow["variant"] = res.basis
        cands = _filter_chunks(chunks, frow)
        idx = BM25Index(cands)
    else:
        cands = chunks
        key = (id(chunks), False)
        if key not in _IDX_CACHE:
            _IDX_CACHE[key] = BM25Index(cands)
        idx = _IDX_CACHE[key]
    ranked = idx.search(row.get("question_text") or row.get("question") or "", k=top_k)
    blocks = []
    for i, score in ranked:
        c = cands[i]
        m = c["meta"]
        blocks.append(
            f"[document_id={m.get('document_id')} page={m.get('page')} "
            f"company={m.get('company_id')} period={m.get('period')} "
            f"variant={m.get('variant')} score={score:.3f}]\n{c['text']}")
    return "\n\n".join(blocks)[:CONTEXT_CHAR_CAP]


def _row_doc_ids(row: dict) -> set[str]:
    ids = set()
    did = row.get("document_id")
    if isinstance(did, str) and did:
        ids.add(did)
    for ev in row.get("evidence") or []:
        if isinstance(ev, dict):
            ed = ev.get("document_id")
            if isinstance(ed, str) and ed:
                ids.add(ed)
    return ids


def _doc_identity_ambiguous(row: dict) -> bool:
    did = row.get("document_id")
    if not isinstance(did, str) or not did.strip():
        ev = row.get("evidence") or []
        if not ev:
            return True
        for e in ev:
            if not isinstance(e, dict) or not e.get("document_id"):
                return True
    return False


def load_few_shot(path: str | Path | None = None,
                  n: int = 3, exclude_id: str | None = None,
                  exclude_doc_ids: set[str] | None = None) -> list[dict]:
    if path is None:
        from src.config import GOLD_PATH
        path = GOLD_PATH
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    dev = [r for r in rows if r.get("split") == "dev"]
    holdout_ids = {str(r.get("id") or r.get("row_id")) for r in rows if r.get("split") == "holdout"}
    out = []
    for r in dev:
        rid = str(r.get("id") or r.get("row_id") or "")
        if rid in holdout_ids:
            raise AssertionError(f"holdout row {rid} in dev candidates")
        if exclude_id is not None and rid == str(exclude_id):
            continue
        if _doc_identity_ambiguous(r):
            continue
        rdocs = _row_doc_ids(r)
        if not rdocs:
            continue
        if exclude_doc_ids and (rdocs & exclude_doc_ids):
            continue
        out.append(r)
        if len(out) >= n:
            break
    return out
