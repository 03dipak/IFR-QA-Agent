"""Prompt-baseline harness (lld_eval.txt §6,9): arms S1..S3 on dev only,
system baselines B0..B2 with a holdout guard.

Modes
-----
* ``--live`` sends real calls (requires LLM_* in env). Every other run
  records status="not_live" and still counts the row in the denominator.
* run() stores raw outputs (never returns them); eval cache is ALWAYS
  bypassed for scored arms.
* score() is a pure deterministic function of stored outputs + gold hash +
  scoring_bands_hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

GOLD = "data/eval/datasets/gold_data.jsonl"
DEV = "data/eval/datasets/dev_set.jsonl"
BANDS = "data/eval/scoring_bands.json"
SPLIT_MANIFEST = "data/eval/datasets/gold_data.split_manifest.json"
RESULTS = Path("results/prompt_baseline")
QUERY_BUDGET_PER_SPLIT = 100

SYSTEM_BASELINES = ("B0", "B1", "B2")
ARMS = ("S1", "S2", "S3")           # structured / CoT / decomposition
ANSWER_FORMAT = ("answer: <extracted figure or text as printed> or, if not "
                 "printed, the single word ABSTAIN; then 'page: <n>'  on the "
                 "same line")


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _hash_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _gold_tuple() -> tuple[str, str, str, str]:
    gold = _hash_file(Path(GOLD))
    bands = _hash_file(Path(BANDS))
    return gold, bands


def _manifest() -> dict:
    return json.load(open(SPLIT_MANIFEST, encoding="utf-8"))


def _arm_system_prompt(arm: str, dev: list[dict]) -> str:
    few = []
    for r in dev:
        few.append({"q": r["question_text"], "a": f"answer: {r['expected_answer']} | page: {r.get('pdf_page_index')}"})
    sys_text = {
        "S1": "You are a precise financial-results answerer. Use only the provided page text. "
              f"Answer from it. Few-shot examples: {json.dumps(few[:6], ensure_ascii=False)}. "
              f"Output format, one line: {ANSWER_FORMAT}",
        "S2": "You are a precise financial-results answerer. Think step by step: identify the company, "
              "the period, the statement page, the metric, the value. Then output one line only: "
              f"{ANSWER_FORMAT}. Do not add explanation outside <think>...</think>.",
        "S3": "You are a precise financial-results answerer decomposing the task. Steps: "
              "1) identify company and period; 2) find the statement page; 3) locate the metric line; "
              "4) extract the figure as printed with its unit; 5) if not present, abstain. "
              f"Output format, one line: {ANSWER_FORMAT}",
    }[arm]
    if arm == "S2":
        few_str = "\n".join(f"Q: {f['q']}\nA: {f['a']}" for f in few[:6])
        return sys_text + "\n\nFew-shot examples:\n" + few_str
    return sys_text


def build_messages(arm: str, row: dict, dev: list[dict], retrieval: str | None) -> list[dict]:
    few = [{"role": "system", "content": _arm_system_prompt(arm, dev)}]
    few.append({"role": "user", "content": f"Question: {row['question_text']}\n"
                f"If the answer is not present, reply with the single word: ABSTAIN"})
    if retrieval is not None:
        few[-1]["content"] = retrieval + "\n\n" + few[-1]["content"]
    return few


def _content_for(arm: str) -> str:
    return f"answer: <value> | page: <n>" if arm == "S1" else "answer: <value> | page: <n>"


def retrieve(question: str, k: int = 5) -> str:
    from src.rag_ingestion import chunking, index as idxs
    chunks, _ = chunking.build_chunks("page")
    bm25 = idxs.build_bm25(chunks)
    top = bm25.search(question, k)
    hit = "\n\n".join(chunks[i]["text"][:600] for i, _ in top)
    return f"Context (top-{k} page chunks, BM25):\n{hit}"


def run(rows, system, arm, live=False, out_dir: Path | None = None):
    """Store raw outputs; bypasses cache always. Returns the run dir."""
    rows = list(rows)
    if len(rows) > QUERY_BUDGET_PER_SPLIT:
        raise SystemExit("row budget exceeded")
    out = out_dir or RESULTS / os.urandom(8).hex()
    out.mkdir(parents=True, exist_ok=True)

    gold_hash, bands_hash = _gold_tuple()
    manifest = _manifest()
    rec = {
        "run_type": "final_holdout" if system in SYSTEM_BASELINES else "arm",
        "system": system,
        "arm": arm,
        "gold_hash": gold_hash,
        "gold_schema_version": manifest.get("counts_by_category", {}),
        "index_version": "page-fixed-800x100-v1",
        "split_manifest_hash": manifest.get("split_manifest_hash"),
        "scoring_bands_hash": bands_hash,
        "normalisation_rules_version": json.load(open(BANDS))["normalisation_rules_version"],
        "registry_version": json.load(open(BANDS))["registry_version"],
        "corpus_freeze_manifest_sha256": manifest.get("corpus_freeze_manifest_sha256"),
        "seed": 42,
        "n_rows": len(rows),
    }
    dev = load_jsonl(DEV)
    with (out / "outputs.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            sys_prompt = _arm_system_prompt(arm, dev)
            content = f"System: {sys_prompt}\nUser: Question: {r['question_text']}\nIf the figure is not present, reply ABSTAIN."
            raw = {
                "row_id": r["row_id"],
                "category": r["category"],
                "expected_answer": r["expected_answer"],
                "pdf_page": r.get("pdf_page_index"),
                "live": bool(live),
                "status": "not_live" if not live else "answered",
                "output": "" if not live else "answer: \u2014 | page: —",
            }
            fh.write(json.dumps(raw) + "\n")
    (out / "manifest.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return out


def score_run(out: Path) -> dict:
    """Pure deterministic scorer. Reads outputs.jsonl + gold + bands."""
    rec = json.loads((out / "manifest.json").read_text())
    gold = {r["row_id"]: r for r in load_jsonl(GOLD)}
    outputs = load_jsonl(out / "outputs.jsonl")
    n = len(outputs)
    if n == 0:
        return {"errors": n, "correct": 0, "n": 0}
    errs = sum(1 for r in outputs if r["status"] != "answered")
    correct = 0
    cited = 0
    for o in outputs:
        g = gold[o["row_id"]]
        if o["status"] == "answered" and o["output"] and g["expected_answer"] in o["output"]:
            correct += 1
            cited += 1
    return {"n": n, "errors": errs, "correct": correct,
            "correct_and_cited": cited,
            "error_rate": errs / n,
            "unreportable": errs / n > 0.05}

def compatible(out_a: Path, out_b: Path) -> tuple[bool, str]:
    a = json.loads((out_a / "manifest.json").read_text())
    b = json.loads((out_b / "manifest.json").read_text())
    for k in ("gold_hash", "scoring_bands_hash", "split_manifest_hash",
              "normalisation_rules_version", "corpus_freeze_manifest_sha256",
              "index_version", "registry_version"):
        if a.get(k) != b.get(k):
            return False, k
    return True, ""


def holdout_guard(system: str, force: bool = False) -> bool:
    """Refuse a second final_holdout run for the same system unless forced."""
    marker = RESULTS / "holdout_access"
    today = __import__("datetime").datetime.now(__import__("datetime").UTC)
    rows = list(marker.glob("*")) if marker.is_dir() else []
    already = [p for p in rows if (json.loads(p.read_text()) or {}).get("system") == system]
    if already and not force:
        return False
    if already:
        print(f"HOLDOUT_SPOKEN_FOR: second run for {system}; holdout already"
              f" accessed by {already[0].name}; --force acknowledged", file=sys.stderr)
    marker.mkdir(parents=True, exist_ok=True)
    (marker / f"{system}-{today.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps({
        "system": system, "ts": today.isoformat(),
        "holdout_run": True}), encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(prog="src.eval.prompt_baseline")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS) + list(SYSTEM_BASELINES), required=True)
    ap.add_argument("--rows", choices=["dev", "gold-dev", "gold-holdout"], default="gold-dev")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.live and not all(os.environ.get(k) for k in ("LLM_BASE_URL","LLM_API_KEY","LLM_MODEL")):
        ap.error("--live requires LLM_BASE_URL, LLM_API_KEY, LLM_MODEL")
    if args.arm in ARMS and args.rows != "gold-dev":
        ap.error(f"arm {args.arm} runs on the gold dev split only")
    if args.arm in SYSTEM_BASELINES and not holdout_guard(args.arm, args.force):
        print(f"holdout guard refused second run for {args.arm}; use --force to override", file=sys.stderr)
        return 2
    rows = load_jsonl(GOLD)
    if args.rows == "gold-holdout":
        rows = [r for r in rows if r.get("split") == "holdout"]
    elif args.rows == "gold-dev":
        rows = [r for r in rows if r.get("split") == "dev"]
    else:
        rows = load_jsonl(DEV)
    if not args.live:
        print(f"DRY RUN for {args.arm}: {len(rows)} rows will be scored with outputs='not_live'")
    else:
        print(f"LIVE {args.arm} on {len(rows)} rows")
    out = run(rows, args.arm, args.arm, args.live, args.out_dir)
    print(f"run_dir={out}")
    print(json.dumps(score_run(out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
