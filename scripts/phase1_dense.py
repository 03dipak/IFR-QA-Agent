import json, time, gc
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from src.eval.retrieval_sanity import answerable_rows, cites, prefix_text

RAW = Path("results/chunks_raw.jsonl")
OUT = Path("results/retrieval_sanity_dense.json")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


def embed_mode(chunks, mode, model):
    texts = [c["text"] if mode == "raw" else prefix_text(c) for c in chunks]
    t0 = time.time()
    emb = list(model.embed(texts, batch_size=256))
    arr = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norm == 0, 1.0, norm)
    log(f"embedded {mode} {len(arr)} in {time.time()-t0:.1f}s "
        f"({len(arr)/(time.time()-t0):.1f}/s)")
    return arr


def recall_for(chunks, arr, model, rows):
    per_cat = {}
    hits = {1: 0, 5: 0, 10: 0}
    n = 0
    for r in rows:
        q = r.get("question_text") or r.get("question") or ""
        qv = np.asarray(next(model.embed([q])), dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) or 1.0)
        scores = arr @ qv
        idx = np.argpartition(-scores, min(10, len(scores)-1))[:10]
        idx = idx[np.argsort(-scores[idx])]
        cit = cites(r)
        cat = r.get("category") or "?"
        d = per_cat.setdefault(cat, {"n": 0, "hits": {1: 0, 5: 0, 10: 0}})
        d["n"] += 1
        n += 1
        for k in (1, 5, 10):
            ok = any((chunks[i]["meta"].get("document_id"),
                      chunks[i]["meta"].get("page")) in cit for i in idx[:k])
            d["hits"][k] += int(ok)
            hits[k] += int(ok)
    out = {"n": n, "recall@1": hits[1]/n if n else 0,
           "recall@5": hits[5]/n if n else 0, "recall@10": hits[10]/n if n else 0,
           "by_category": {}}
    for cat, d in sorted(per_cat.items()):
        out["by_category"][cat] = {
            f"recall@{k}": d["hits"][k]/d["n"] if d["n"] else 0, "n": d["n"]}
    return out


def main():
    t0 = time.time()
    chunks = load_page_chunks(RAW)
    log(f"loaded page chunks {len(chunks)}")
    model = TextEmbedding(model_name=MODEL)
    res = {"note": "page chunks only; dense all-MiniLM-L6-v2"}
    for mode in ("raw", "prefixed"):
        arr = embed_mode(chunks, mode, model)
        np.save(Path(f"results/dense_{mode}.npy"), arr)
        log(f"saved dense_{mode}.npy")
        res[mode] = recall_for(chunks, arr, model, answerable_rows())
        log(f"{mode} recall {res[mode]['recall@1']:.3f}/{res[mode]['recall@5']:.3f}/{res[mode]['recall@10']:.3f}")
        del arr
        gc.collect()
    res["elapsed_seconds"] = round(time.time()-t0, 1)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"WROTE {OUT} elapsed={res['elapsed_seconds']}")


if __name__ == "__main__":
    main()
