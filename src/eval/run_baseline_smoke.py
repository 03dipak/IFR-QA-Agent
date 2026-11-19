"""Smoke run of the retrieval + answer baselines on the DRAFT candidate rows.

Output is labelled DRAFT, NOT GOLD (candidates.csv rows are unverified drafts).
Writes results/baseline_smoke.json with the config tuple.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from src.rag_ingestion import chunking
from src.rag_ingestion import index as idx
from src.eval import retrieval as retr
from src.eval import answer_scoring as ans

ROWS = "data/gold/candidates.csv"
OUT = "results/baseline_smoke.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def gold_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retrieve_for_row(index, query: str, k: int = 10):
    return [index.chunks[i] for i, _ in index.search(query, k)]


def main() -> int:
    ap = argparse.ArgumentParser(prog="src.eval.run_baseline_smoke")
    ap.add_argument("--rows", default=ROWS)
    ap.add_argument("--output", default=OUT)
    ap.add_argument("--top-k", type=int, default=5, help="top-k chunks for the answer prompt")
    ap.add_argument("--no-dense", action="store_true")
    args = ap.parse_args()

    rows = retr.load_rows(args.rows)
    t0 = time.time()
    results: dict = {
        "note": "DRAFT ROWS, NOT GOLD — recall computed against draft citations",
        "rows_file": args.rows,
        "rows_gold_sha256": gold_sha256(args.rows),
        "n_rows": len(rows),
        "slice": {"annuals_fy_2024_2026": True, "dense_quarterlies_of": list(chunking.DENSE_COMPANIES)},
        "config": {},
        "retrieval": {},
        "answer_baseline": {},
    }
    for variant in ("page", "fixed"):
        print(f"building chunks variant={variant} ...", flush=True)
        chunks, chunk_secs = chunking.build_chunks(variant)
        bm25 = idx.build_bm25(chunks)
        entry = {
            "chunks": len(chunks),
            "chunk_build_seconds": round(chunk_secs, 2),
            "bm25_build_seconds": round(bm25.build_seconds, 2),
            "bm25": {},
            "dense": None,
        }
        for mode, index in (("bm25", bm25),):
            retrieved = [retrieve_for_row(index, r["question"], 10) for r in rows]
            entry[mode] = retr.evaluate(rows, retrieved, (1, 3, 5, 10))
            print(f"variant={variant} mode={mode} {entry[mode]}", flush=True)
        if not args.no_dense:
            try:
                dense = idx.DenseIndex(chunks, EMBED_MODEL)
                entry["dense_build_seconds"] = round(dense.build_seconds, 2)
                retrieved = [retrieve_for_row(dense, r["question"], 10) for r in rows]
                entry["dense"] = retr.evaluate(rows, retrieved, (1, 3, 5, 10))
                print(f"variant={variant} mode=dense {entry['dense']}", flush=True)
            except Exception as e:
                entry["dense"] = {"status": "skipped", "reason": f"{type(e).__name__}: {e}"[:200]}
                print(f"dense skipped: {entry['dense']['reason']}", flush=True)
        else:
            entry["dense"] = {"status": "skipped", "reason": "--no-dense"}
        entry["index_version_bm25"] = chunking.index_version(variant, "bm25-own")
        entry["index_version_dense"] = chunking.index_version(variant, EMBED_MODEL)
        results["retrieval"][variant] = entry

    results["config"] = {
        "index_version_page": chunking.index_version("page", EMBED_MODEL),
        "index_version_fixed": chunking.index_version("fixed", EMBED_MODEL),
        "chunk_config": chunking.CHUNK_CONFIG,
        "embedding_model": EMBED_MODEL,
        "freeze_manifest_sha256": chunking.freeze_manifest_sha256(),
        "rows_gold_sha256": results["rows_gold_sha256"],
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # Answer baseline (skipped unless an endpoint is configured)
    if not ans.endpoint_configured():
        results["answer_baseline"] = {
            "status": "skipped",
            "reason": "no endpoint configured: set OPENAI_BASE_URL, "
                      "OPENAI_API_KEY, OPENAI_MODEL (hosted <=8B "
                      "OpenAI-compatible)",
            "rows_unscored": len(rows),
        }
    else:
        dense_index = idx.DenseIndex(chunking.build_chunks("page")[0], EMBED_MODEL)
        per_row = []
        for r in rows:
            chunks_for_row = retrieve_for_row(dense_index, r["question"], args.top_k)
            messages = ans.single_shot_prompt(r["question"], chunks_for_row)
            try:
                out = ans.call_endpoint(messages)
            except Exception as e:
                per_row.append({"category": r["category"], "error": f"{type(e).__name__}"})
                continue
            per_row.append({"category": r["category"],
                            **ans.score_answer(r, out)})
        scored = [p for p in per_row if "numeric_match" in p or p.get("abstain_correct") is not None]
        results["answer_baseline"] = {
            "status": "run",
            "top_k": args.top_k,
            "n": len(rows),
            "numeric_match": sum(1 for p in scored if p.get("numeric_match")) / max(1, sum(1 for p in scored if p.get("numeric_match") is not None)),
            "abstain_correct": sum(1 for p in scored if p.get("abstain_correct")) / max(1, len(scored)),
            "citation_match": sum(1 for p in scored if p.get("citation_match")) / max(1, len(scored)),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
