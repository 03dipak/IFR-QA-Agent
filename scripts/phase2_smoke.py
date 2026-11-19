import json, time
from pathlib import Path
from src.eval.retrieval_sanity import answerable_rows
from src.eval.contexts import oracle_context, rag_context, load_few_shot
from src.eval.baselines import run_b0, run_b1, run_b2

OUT = Path("results/phase2_smoke.json")
CHUNKS = Path("results/chunks_raw.jsonl")


def main():
    rows = answerable_rows()[:5]
    few = load_few_shot(n=3)
    chunks = []
    with CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("meta", {}).get("chunk_type") == "page":
                chunks.append(c)
    results = []
    for r in rows:
        for label, fn in (
            ("B0", lambda: run_b0(r, few_shot_rows=few, live=True)),
            ("B1", lambda: run_b1(r, context=oracle_context(r), live=True)),
            ("B2-global", lambda: run_b2(r, context=rag_context(r, chunks, filtered=False),
                                         retrieval_variant="global", live=True)),
            ("B2-filtered", lambda: run_b2(r, context=rag_context(r, chunks, filtered=True),
                                           retrieval_variant="filtered", live=True)),
        ):
            try:
                results.append(fn())
            except Exception as e:  # smoke: record and continue
                results.append({"row_id": r.get("row_id") or r.get("id"),
                                "baseline": label.split("-")[0],
                                "retrieval_variant": label.split("-")[-1].lower(),
                                "status": f"error: {type(e).__name__}: {e}"[:300]})
                print("ERR", r.get("row_id") or r.get("id"), label, type(e).__name__, e, flush=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SMOKE_NOT_SCORED", OUT)
    for x in results:
        print(x["row_id"], x["baseline"], x["retrieval_variant"],
              "prompt_tokens=", x.get("prompt_tokens"), "completion_tokens=", x.get("completion_tokens"),
              "latency_ms=", x.get("latency_ms"), "status=", x.get("status"),
              "output=", repr((x.get("output") or "")[:80]))


if __name__ == "__main__":
    main()
