"""Retrieval sanity measurements (Phase 1).

Oracle and global recall@k for BM25 (and dense, via background job) using the
page-level gold evidence hit definition. No scored runs, no gold edits.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.rag_ingestion import chunking
from src.rag_ingestion.index import BM25Index
from src.eval.retrieval import hit_at_k

FREEZE = Path("data/corpus/freeze_v1.json")
PAGE_DIR = Path("data/corpus/extract/page_extracts")
TABLE_DIR = Path("data/corpus/extract/table_records")
GOLD = Path("data/eval/datasets/gold_data.jsonl")


def freeze_docs() -> list[dict]:
    return json.loads(FREEZE.read_text(encoding="utf-8"))["documents"]


def page_usable(doc_id: str, page: int) -> bool:
    path = PAGE_DIR / f"{doc_id}.pages.jsonl"
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("pdf_page_index") == page:
            return bool(rec.get("usable_as_evidence"))
    return False


def _doc_by_id() -> dict[str, dict]:
    return {d["document_id"]: d for d in freeze_docs()}


def build_all_chunks(text_mode: str = "raw") -> list[dict]:
    """Chunks over every freeze document, usable pages only."""
    chunks: list[dict] = []
    for doc in freeze_docs():
        doc_id = doc["document_id"]
        # Read page records once; page chunks and the usable set share it.
        page_path = PAGE_DIR / f"{doc_id}.pages.jsonl"
        pages: list[dict] = []
        if page_path.is_file():
            for line in page_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    pages.append(json.loads(line))
        usable = {r.get("pdf_page_index") for r in pages
                  if r.get("usable_as_evidence") and isinstance(r.get("pdf_page_index"), int)}
        for rec in pages:
            page = rec.get("pdf_page_index")
            if not isinstance(page, int) or page not in usable:
                continue
            text = rec.get("page_text") or ""
            if not text.strip():
                continue
            meta = {
                "document_id": doc_id,
                "page": int(page),
                "company_id": doc.get("company_id"),
                "period": doc.get("fiscal_year"),
                "variant": doc.get("variant"),
                "doc_type": doc.get("doc_type"),
                "statement_type": doc.get("doc_type"),
                "chunk_type": "page",
            }
            chunk_text = text
            if text_mode == "prefixed":
                st = doc.get("doc_type")
                chunk_text = (f"{meta['company_id']} | {meta['period']} | "
                              f"{meta['variant']} | {st} | {meta['page']}\n{text}")
            chunks.append({"id": f"{doc_id}:p{page}", "text": chunk_text, "meta": meta})

        # table chunks
        table_path = TABLE_DIR / f"{doc_id}.tables.jsonl"
        if table_path.is_file():
            for n, line in enumerate(table_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                rec = json.loads(line)
                page = rec.get("page_start")
                if not isinstance(page, int) or page not in usable:
                    continue
                parts = [f"statement_kind: {rec.get('statement_kind')}",
                         f"unit: {(rec.get('unit_header') or '').strip()}",
                         f"caption: {(rec.get('caption') or '').strip()}",
                         f"reporting_basis: {rec.get('reporting_basis')}"]
                rows = rec.get("rows") or []
                for r in rows:
                    rl = (r.get("row_label_raw") or "").strip()
                    cells = r.get("cells") or []
                    raw = [str(c.get("raw") or "").strip() for c in cells]
                    raw = [c for c in raw if c]
                    parts.append(f"{rl}: {' | '.join(raw)}" if rl else " | ".join(raw))
                text = "\n".join(p for p in parts if p.strip())
                if not text.strip():
                    continue
                st = rec.get("statement_kind") or doc.get("doc_type")
                meta = {
                    "document_id": doc_id,
                    "page": int(page),
                    "company_id": doc.get("company_id"),
                    "period": doc.get("fiscal_year"),
                    "variant": doc.get("variant"),
                    "doc_type": doc.get("doc_type"),
                    "statement_kind": rec.get("statement_kind"),
                    "statement_type": st,
                    "chunk_type": "table",
                }
                chunk_text = text
                if text_mode == "prefixed":
                    chunk_text = (f"{meta['company_id']} | {meta['period']} | "
                                  f"{meta['variant']} | {st} | {meta['page']}\n{text}")
                chunks.append({"id": f"{doc_id}:t{n}", "text": chunk_text, "meta": meta})
    chunks.sort(key=lambda c: c["id"])
    return chunks


def prefix_text(chunk: dict) -> str:
    m = chunk["meta"]
    st = m.get("statement_type") or m.get("doc_type") or m.get("statement_kind")
    return (f"{m.get('company_id')} | {m.get('period')} | {m.get('variant')} | "
            f"{st} | {m.get('page')}\n{chunk['text']}")


def with_prefix(chunks: list[dict]) -> list[dict]:
    return [{"id": c["id"], "text": prefix_text(c), "meta": c["meta"]} for c in chunks]


def answerable_rows() -> list[dict]:
    rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("status") == "active"
            and r.get("expected_behaviour") == "answer"]


def cites(row: dict) -> set[tuple[str, int]]:
    out = set()
    for e in row.get("evidence") or []:
        did = e.get("document_id")
        pg = e.get("pdf_page")
        if isinstance(did, str) and isinstance(pg, int):
            out.add((did, pg))
    # fallback top-level
    if not out and row.get("document_id") and isinstance(row.get("pdf_page_index"), int):
        out.add((row["document_id"], row["pdf_page_index"]))
    return out


def oracle_index_check() -> list[dict]:
    problems = []
    for r in answerable_rows():
        for (did, pg) in cites(r):
            if not page_usable(did, pg):
                problems.append({"row_id": r.get("row_id") or r.get("id"),
                                 "document_id": did, "pdf_page": pg,
                                 "reason": "not_in_index_or_not_usable"})
    return problems


def _filter_chunks(chunks: list[dict], row: dict) -> list[dict]:
    companies = [c for c in str(row.get("company_id") or "").split("|") if c]
    cset = set(companies)
    period = row.get("fiscal_year") or row.get("period")
    basis = row.get("variant")
    doc_type = row.get("doc_type")
    out = []
    for ch in chunks:
        m = ch["meta"]
        if cset and m.get("company_id") not in cset:
            continue
        if period and m.get("period") != period:
            continue
        if basis and m.get("variant") != basis:
            continue
        if doc_type and m.get("doc_type") != doc_type:
            continue
        out.append(ch)
    return out


def _recall_for(chunks: list[dict], rows: list[dict], filtered: bool) -> dict:
    # group chunks for filtered indexes
    cache: dict[tuple, BM25Index] = {}
    per_cat: dict[str, dict[int, list[bool]]] = defaultdict(lambda: {1: [], 5: [], 10: []})
    hits_by_k = {1: 0, 5: 0, 10: 0}
    n = 0
    for row in rows:
        if filtered:
            key = (tuple(sorted(str(row.get("company_id") or "").split("|"))),
                   row.get("fiscal_year") or row.get("period"),
                   row.get("variant"), row.get("doc_type"))
            if key not in cache:
                cands = _filter_chunks(chunks, row)
                cache[key] = BM25Index(cands) if cands else None
            idx = cache[key]
            cand_chunks = None
            # retrieve from filtered candidate list
            if idx is None:
                retrieved = []
            else:
                # need map ranked indices back to chunks; BM25Index.chunks is candidate list
                ranked = idx.search(row.get("question_text") or row.get("question") or "", k=10)
                retrieved = [idx.chunks[i] for i, _ in ranked]
        else:
            if None not in cache:
                cache[None] = BM25Index(chunks)
            idx = cache[None]
            ranked = idx.search(row.get("question_text") or row.get("question") or "", k=10)
            retrieved = [idx.chunks[i] for i, _ in ranked]
        cit = cites(row)
        cat = row.get("category") or row.get("bucket") or "?"
        n += 1
        for k in (1, 5, 10):
            h = hit_at_k(retrieved, cit, k)
            per_cat[cat][k].append(h)
            if h:
                hits_by_k[k] += 1
    out = {"n": n}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = hits_by_k[k] / n if n else 0.0
        out[f"hits@{k}"] = hits_by_k[k]
    out["by_category"] = {}
    for cat, d in sorted(per_cat.items()):
        out["by_category"][cat] = {}
        for k in (1, 5, 10):
            vals = d[k]
            out["by_category"][cat][f"recall@{k}"] = (sum(vals) / len(vals)) if vals else 0.0
            out["by_category"][cat][f"n"] = len(vals)
    return out


def run_phase1_bm25() -> dict:
    rows = answerable_rows()
    res = {}
    idx_problems = oracle_index_check()
    res["oracle_index_problems"] = idx_problems
    res["n_answerable"] = len(rows)
    for mode in ("raw", "prefixed"):
        chunks = build_all_chunks(mode)
        res[f"filtered_{mode}"] = _recall_for(chunks, rows, filtered=True)
        res[f"global_{mode}"] = _recall_for(chunks, rows, filtered=False)
    # raw vs prefixed delta (global)
    res["global_delta_prefixed_minus_raw"] = {
        f"recall@{k}": res[f"global_prefixed"][f"recall@{k}"] - res[f"global_raw"][f"recall@{k}"]
        for k in (1, 5, 10)
    }
    out_path = Path("results/retrieval_sanity_bm25.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
