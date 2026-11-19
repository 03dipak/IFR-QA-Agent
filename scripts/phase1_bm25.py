import json, time, gc
from pathlib import Path
from src.eval.retrieval_sanity import answerable_rows, oracle_index_check, _recall_for

OUT = Path("results/retrieval_sanity_bm25.json")
FILES = {"raw": Path("results/chunks_raw.jsonl"),
         "prefixed": Path("results/chunks_prefixed.jsonl")}


def log(msg):
    print(msg, flush=True)


def load_page_chunks(p):
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            c = json.loads(line)
            if c.get("meta", {}).get("chunk_type") == "page":
                out.append(c)
    return out


def main():
    t0 = time.time()
    rows = answerable_rows()
    res = {"oracle_index_problems": oracle_index_check(), "n_answerable": len(rows),
           "note": "page chunks only; table chunks omitted under 4.8GB memory ceiling"}
    for mode, path in FILES.items():
        t = time.time()
        chunks = load_page_chunks(path)
        log(f"loaded {mode} page chunks {len(chunks)} in {time.time()-t:.1f}s")
        res[f"filtered_{mode}"] = _recall_for(chunks, rows, filtered=True)
        log(f"filtered_{mode} {time.time()-t:.1f}s")
        res[f"global_{mode}"] = _recall_for(chunks, rows, filtered=False)
        log(f"global_{mode} {time.time()-t:.1f}s")
        del chunks
        gc.collect()
    res["global_delta_prefixed_minus_raw"] = {
        f"recall@{k}": res["global_prefixed"][f"recall@{k}"] - res["global_raw"][f"recall@{k}"]
        for k in (1, 5, 10)
    }
    res["elapsed_seconds"] = round(time.time()-t0, 1)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"WROTE {OUT} elapsed={res['elapsed_seconds']}")


if __name__ == "__main__":
    main()
