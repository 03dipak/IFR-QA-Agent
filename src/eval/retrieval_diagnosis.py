"""Day 4 retrieval diagnosis: BM25/dense/RRF traces and root-cause evidence (LLD-29).

Diagnosis-only. Reads existing artifacts, frozen gold/index/page extracts, and the
existing Retriever without modifying retrieval, prompts, gold, or orchestration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.eval import agent_eval
from src.rag_ingestion.contract import read_page_extracts

CHUNK_RE = re.compile(r"^(?P<doc>.+)::C2::p(?P<page>\d+)::w(?P<win>\d+)$")
WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return WS.sub(" ", (text or "").lower()).strip()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gold_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in agent_eval.load_gold(path)}


def page_map(storage_dir: str, doc_id: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for rec in read_page_extracts(storage_dir, doc_id):
        page = rec.get("pdf_page_index")
        if isinstance(page, int):
            out[page] = rec
    return out


def find_excerpt_page(pages: dict[int, dict[str, Any]], excerpt: str) -> list[int]:
    needle = norm(excerpt)
    if not needle:
        return []
    return [p for p, rec in sorted(pages.items()) if needle in norm(rec.get("page_text") or "")]


def full_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=ranks.dtype)
    return ranks


def top_rows(scores: np.ndarray, ids: list[str], topn: int) -> list[dict[str, Any]]:
    order = np.argsort(-scores, kind="mergesort")[:topn]
    rows = []
    for rank, idx in enumerate(order, start=1):
        match = CHUNK_RE.match(ids[int(idx)])
        rows.append({
            "rank": rank, "chunk_index": int(idx), "score": float(scores[idx]),
            "doc_id": match.group("doc")[:12] if match else "",
            "page": int(match.group("page")) if match else None,
            "chunk_id": ids[int(idx)],
        })
    return rows


def expected_hits(
    ranks: np.ndarray, scores: np.ndarray, ids: list[str], doc: str, page: int,
) -> list[dict[str, Any]]:
    hits = []
    for idx, cid in enumerate(ids):
        match = CHUNK_RE.match(cid)
        if match and match.group("doc")[:12] == doc and int(match.group("page")) == page:
            hits.append({"rank": int(ranks[idx]), "score": float(scores[idx]), "chunk_id": cid})
    hits.sort(key=lambda row: row["rank"])
    return hits


def best_doc(
    ranks: np.ndarray, scores: np.ndarray, ids: list[str], doc: str,
) -> dict[str, Any] | None:
    best = None
    for idx, cid in enumerate(ids):
        match = CHUNK_RE.match(cid)
        if match and match.group("doc")[:12] == doc:
            row = {"rank": int(ranks[idx]), "score": float(scores[idx]),
                   "page": int(match.group("page")), "chunk_id": cid}
            if best is None or row["rank"] < best["rank"]:
                best = row
    return best


def rrf_scores(retriever: Any, bm25: np.ndarray, dense: np.ndarray) -> np.ndarray:
    pool = min(retriever.pool, len(bm25))
    bidx = np.argpartition(-bm25, pool - 1)[:pool]
    bidx = bidx[np.argsort(-bm25[bidx], kind="mergesort")]
    didx = np.argpartition(-dense, pool - 1)[:pool]
    didx = didx[np.argsort(-dense[didx], kind="mergesort")]
    out = np.zeros(len(bm25), dtype=np.float32)
    k = retriever.k
    for rank, idx in enumerate(bidx, start=1):
        out[int(idx)] += 1.0 / (k + rank)
    for rank, idx in enumerate(didx, start=1):
        out[int(idx)] += 1.0 / (k + rank)
    return out


def stages_for_query(retriever: Any, query: str, topn: int = 10) -> dict[str, Any]:
    from src.tools.retrieve import cleaned_tokens, embed_query

    tokens = cleaned_tokens(query)
    bm25_scores = np.asarray(retriever._bm25().get_scores(tokens), dtype=np.float32)
    qvec = embed_query(query, config=retriever.config)
    q = qvec.astype(np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    dense_scores = (retriever.embeddings @ q).astype(np.float32)
    rrf = rrf_scores(retriever, bm25_scores, dense_scores)
    bm25_r = full_ranks(bm25_scores)
    dense_r = full_ranks(dense_scores)
    rrf_r = full_ranks(rrf)
    pool = np.argsort(-rrf, kind="mergesort")[: retriever.pool]
    reranked = retriever._rerank(query, [(int(i), float(rrf[i])) for i in pool])
    final = []
    for rank, (idx, score) in enumerate(reranked, start=1):
        match = CHUNK_RE.match(retriever.ids[idx])
        final.append({
            "rank": rank, "chunk_index": int(idx), "score": float(score),
            "doc_id": match.group("doc")[:12] if match else "",
            "page": int(match.group("page")) if match else None,
            "chunk_id": retriever.ids[idx],
        })
    return {
        "query": query, "tokens": tokens,
        "bm25_top": top_rows(bm25_scores, retriever.ids, topn),
        "dense_top": top_rows(dense_scores, retriever.ids, topn),
        "rrf_top": top_rows(rrf, retriever.ids, topn),
        "final_top": final,
        "_bm25": bm25_scores, "_dense": dense_scores, "_rrf": rrf,
        "_bm25_r": bm25_r, "_dense_r": dense_r, "_rrf_r": rrf_r,
    }


def evidence_check(gold_row: dict[str, Any], storage_dir: str) -> list[dict[str, Any]]:
    checks = []
    for ev in gold_row.get("evidence", []):
        doc = str(ev.get("document_id") or "")
        page = ev.get("pdf_page")
        excerpt = str(ev.get("excerpt") or "")
        pages = page_map(storage_dir, doc)
        rec = pages.get(page) if isinstance(page, int) else None
        text = rec.get("page_text") or "" if rec else ""
        found = find_excerpt_page(pages, excerpt)
        checks.append({
            "doc_id": doc[:12], "page": page,
            "page_exists": page in pages if isinstance(page, int) else False,
            "excerpt_in_page": norm(excerpt) in norm(text) if excerpt and text else False,
            "excerpt_found_pages": found[:5],
            "page_text_len": len(text),
        })
    return checks


def _in_pool(hits: list[dict[str, Any]], pool: int) -> list[dict[str, Any]]:
    return [h for h in hits if int(h["rank"]) <= pool]


def classify(expected_doc: str, expected_page: int, stage_info: dict[str, Any],
             evidence: dict[str, Any], *, pool: int, topk: int) -> tuple[str, str]:
    ids = stage_info["ids"]
    bm25 = expected_hits(stage_info["_bm25_r"], stage_info["_bm25"], ids, expected_doc,
        expected_page)
    dense = expected_hits(stage_info["_dense_r"], stage_info["_dense"], ids, expected_doc,
        expected_page)
    rrf = expected_hits(stage_info["_rrf_r"], stage_info["_rrf"], ids, expected_doc, expected_page)
    final = [r for r in stage_info["final_top"]
             if r["doc_id"] == expected_doc and r["page"] == expected_page]
    doc_bm25 = best_doc(stage_info["_bm25_r"], stage_info["_bm25"], ids, expected_doc)
    doc_dense = best_doc(stage_info["_dense_r"], stage_info["_dense"], ids, expected_doc)
    doc_rrf = best_doc(stage_info["_rrf_r"], stage_info["_rrf"], ids, expected_doc)
    b_in, d_in, r_in = _in_pool(bm25, pool), _in_pool(dense, pool), _in_pool(rrf, pool)
    f_in = [r for r in final if int(r["rank"]) <= topk]

    if not evidence.get("page_exists", True):
        return "page_metadata_inconsistency", "gold page missing from page extracts"
    found_pages = evidence.get("excerpt_found_pages", [])
    if found_pages and expected_page not in found_pages:
        return "pdf_page_number_offset", f"excerpt found on {found_pages[:3]} not expected page"
    if evidence.get("page_exists") and not evidence.get("excerpt_in_page", True):
        return "page_metadata_inconsistency", "gold excerpt not found on expected page text"

    if f_in:
        return "other_insufficient_evidence", "expected page present in final top-k"
    if r_in:
        return ("reranker_dropped_expected_page",
                "expected page in RRF top-pool but absent from final top-k")
    if b_in or d_in:
        return ("rrf_fusion_loss",
                "expected page in BM25/dense top-pool but absent from RRF top-pool")
    if doc_rrf and int(doc_rrf["rank"]) <= pool:
        return ("correct_doc_wrong_page_ranking",
                "expected doc in RRF top-pool but expected page not in top-pool")
    if ((doc_bm25 and int(doc_bm25["rank"]) <= pool)
            or (doc_dense and int(doc_dense["rank"]) <= pool)):
        return ("correct_doc_wrong_page_ranking",
                "expected doc in BM25/dense top-pool but expected page not in top-pool")
    if doc_bm25 or doc_dense or doc_rrf:
        return ("query_document_selection_failure",
                "expected doc present only outside retrieval pool; query/doc selection weak")
    return ("query_document_selection_failure",
            "expected doc/page absent from BM25 and dense rankings")


def build_report(run_dir: Path, day3_dir: Path, gold_path: Path, out_dir: Path,
                 storage_dir: str = "data/corpus") -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from src import config  # noqa: F401
    from src.tools.retrieve import Retriever

    retriever = Retriever()
    ids = retriever.ids
    gold = gold_map(gold_path)
    d_rows = read_rows(day3_dir / "table_d_failure_classification.csv")
    traces_dir = Path("logs/traces")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]

    stage_cache: dict[str, dict[str, Any]] = {}
    query_traces = []
    failure_rows = []
    comparison_rows = []

    for d in d_rows:
        row_id = d["query_id"]
        if "wrong_document_retrieved" not in d["classifications"] and \
           "correct_document_wrong_page" not in d["classifications"]:
            continue
        events = [json.loads(line) for line in
                  (traces_dir / f"eval_{run_id}_{row_id}.jsonl").read_text(encoding="utf-8")
                  .splitlines() if line.strip()]
        calls = [e for e in events if e.get("stage") == "tool" and e.get("tool") == "retrieve"]
        gold_row = gold[row_id]
        checks = {f"{c['doc_id']}:{c['page']}": c for c in evidence_check(gold_row, storage_dir)}
        gold_pairs = [(str(ev["document_id"])[:12], int(ev["pdf_page"]))
                      for ev in gold_row.get("evidence", [])]

        for call_index, call in enumerate(calls):
            query = str((call.get("args") or {}).get("query") or "")
            if query not in stage_cache:
                stage_cache[query] = stages_for_query(retriever, query)
            st = stage_cache[query]
            st["ids"] = ids
            query_traces.append({
                "row_id": row_id, "call_index": call_index, "query": query,
                "tokens": st["tokens"],
                "bm25_top": st["bm25_top"], "dense_top": st["dense_top"],
                "rrf_top": st["rrf_top"], "final_top": st["final_top"],
            })
            for expected_doc, expected_page in gold_pairs:
                bm25_hits = expected_hits(st["_bm25_r"], st["_bm25"], ids, expected_doc,
                    expected_page)
                dense_hits = expected_hits(st["_dense_r"], st["_dense"], ids, expected_doc,
                    expected_page)
                rrf_hits = expected_hits(st["_rrf_r"], st["_rrf"], ids, expected_doc, expected_page)
                final_hits = [r for r in st["final_top"]
                              if r["doc_id"] == expected_doc and r["page"] == expected_page]
                doc_bm25 = best_doc(st["_bm25_r"], st["_bm25"], ids, expected_doc)
                doc_dense = best_doc(st["_dense_r"], st["_dense"], ids, expected_doc)
                doc_rrf = best_doc(st["_rrf_r"], st["_rrf"], ids, expected_doc)
                ev = checks.get(f"{expected_doc}:{expected_page}", {})
                rc, reason = classify(
                    expected_doc, expected_page, st, ev,
                    pool=retriever.pool, topk=retriever.top_k,
                )
                comparison_rows.append({
                    "row_id": row_id, "call_index": call_index,
                    "expected_doc": expected_doc, "expected_page": expected_page,
                    "bm25_best_rank": bm25_hits[0]["rank"] if bm25_hits else None,
                    "bm25_best_score": bm25_hits[0]["score"] if bm25_hits else None,
                    "dense_best_rank": dense_hits[0]["rank"] if dense_hits else None,
                    "dense_best_score": dense_hits[0]["score"] if dense_hits else None,
                    "rrf_best_rank": rrf_hits[0]["rank"] if rrf_hits else None,
                    "rrf_best_score": rrf_hits[0]["score"] if rrf_hits else None,
                    "final_best_rank": final_hits[0]["rank"] if final_hits else None,
                    "final_best_score": final_hits[0]["score"] if final_hits else None,
                    "bm25_doc_best_rank": doc_bm25["rank"] if doc_bm25 else None,
                    "dense_doc_best_rank": doc_dense["rank"] if doc_dense else None,
                    "rrf_doc_best_rank": doc_rrf["rank"] if doc_rrf else None,
                    "page_exists": ev.get("page_exists"),
                    "excerpt_in_page": ev.get("excerpt_in_page"),
                    "excerpt_found_pages": "|".join(map(str, ev.get("excerpt_found_pages", []))),
                    "root_cause": rc, "reason": reason,
                })
                failure_rows.append(comparison_rows[-1] | {
                    "query": query,
                    "bm25_top5_docs": "|".join(f"{r['doc_id']}:{r['page']}" for r in 
                        st["bm25_top"][:5]),
                    "dense_top5_docs": "|".join(f"{r['doc_id']}:{r['page']}" for r in 
                        st["dense_top"][:5]),
                    "rrf_top5_docs": "|".join(f"{r['doc_id']}:{r['page']}" for r in 
                        st["rrf_top"][:5]),
                    "final_top5_docs": "|".join(f"{r['doc_id']}:{r['page']}" for r in 
                        st["final_top"][:5]),
                })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["row_id", "call_index", "expected_doc", "expected_page", "query",
            "bm25_best_rank", "bm25_best_score", "dense_best_rank", "dense_best_score",
            "rrf_best_rank", "rrf_best_score", "final_best_rank", "final_best_score",
            "bm25_doc_best_rank", "dense_doc_best_rank", "rrf_doc_best_rank",
            "page_exists", "excerpt_in_page", "excerpt_found_pages",
            "bm25_top5_docs", "dense_top5_docs", "rrf_top5_docs", "final_top5_docs",
            "root_cause", "reason"]
    write_csv(out_dir / "failure_analysis.csv", failure_rows, cols)
    comp_cols = [c for c in cols if c not in {"query", "bm25_top5_docs", "dense_top5_docs",
                                               "rrf_top5_docs", "final_top5_docs"}]
    write_csv(out_dir / "bm25_dense_rrf_comparison.csv", comparison_rows, comp_cols)
    (out_dir / "query_traces.json").write_text(json.dumps(query_traces, indent=2), encoding="utf-8")
    unresolved = citation_unresolved_analysis(run_dir, day3_dir)
    summary = summarize(failure_rows, unresolved)
    (out_dir / "unresolved_citations.json").write_text(json.dumps(unresolved, indent=2),
        encoding="utf-8")
    (out_dir / "root_cause_summary.json").write_text(json.dumps(summary, indent=2),
        encoding="utf-8")
    return {"summary": summary, "failure_rows": failure_rows, "query_traces": query_traces,
            "unresolved": unresolved}


def citation_unresolved_analysis(run_dir: Path, day3_dir: Path) -> list[dict[str, Any]]:
    per_query = [
        json.loads(line)
        for line in (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    c_rows = read_rows(day3_dir / "table_c_citations.csv")
    out = []
    for row in per_query:
        if row.get("stopped_reason") != "citation_unresolved":
            continue
        rid = row["row_id"]
        cites = [c for c in c_rows if c["query_id"] == rid]
        has_diag = any(c["generated_citation"] != "(none)" for c in cites)
        parsed = any(c.get("harness_citation_parsed") == "yes" for c in cites)
        if has_diag and not parsed:
            cause = "citation_parser_format_mismatch"
            detail = ("answer contains citation-like spans but production/harness parser "
                      "accepted none")
        elif not has_diag:
            cause = "generation_missing_citation"
            detail = "answer contains no citation-like span; agent stopped unresolved"
        else:
            cause = "citation_evidence_mapping_failure"
            detail = "citations parsed but unresolved against retrieved evidence"
        out.append({
            "row_id": rid, "stopped_reason": row["stopped_reason"],
            "answer_correct": row.get("answer_correct"),
            "diagnostic_citations": sum(c["generated_citation"] != "(none)" for c in cites),
            "harness_parsed_citations": sum(c.get("harness_citation_parsed") == "yes" for c in 
                cites),
            "root_cause": cause, "evidence": detail,
        })
    return out


def summarize(rows: list[dict[str, Any]], unresolved: list[dict[str, Any]]) -> dict[str, Any]:
    by_cause: dict[str, int] = {}
    by_row: dict[str, set[str]] = {}
    for row in rows:
        by_cause[row["root_cause"]] = by_cause.get(row["root_cause"], 0) + 1
        by_row.setdefault(row["row_id"], set()).add(row["root_cause"])
    return {
        "n_failure_rows": len(rows),
        "n_queries": len({(r["row_id"], r["call_index"]) for r in rows}),
        "cause_counts": by_cause,
        "row_causes": {k: sorted(v) for k, v in sorted(by_row.items())},
        "unresolved_citations": unresolved,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("results/evaluation/day2_fix_v1"))
    parser.add_argument("--day3-dir", type=Path, default=Path("results/day3_evidence_v1"))
    parser.add_argument("--gold", type=Path, default=agent_eval.GOLD)
    parser.add_argument("--out-dir", type=Path, default=Path("results/day4_retrieval_diagnosis"))
    args = parser.parse_args()
    result = build_report(args.run_dir, args.day3_dir, args.gold, args.out_dir)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
