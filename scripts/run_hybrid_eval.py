import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

ROOT = Path("/home/dipak/agentic/IFR-QA-Agent")
FIELDS = [
    "row_id", "config", "question", "expected_behaviour", "correct",
    "hit_at_k", "tools", "latency_s", "citation_present", "stop_reason",
    "answer", "citations",
]


def _gold_evidence(row):
    evidence = row.get("evidence") or []
    if not evidence:
        return None, None
    entry = evidence[0]
    return entry.get("document_id"), entry.get("pdf_page") or entry.get("pdf_page_index")


def _is_correct(row, result):
    behaviour = row.get("expected_behaviour")
    answer = (result.answer or "").lower()
    if behaviour == "abstain":
        return any(
            token in answer
            for token in ("insufficient", "not found", "cannot", "refuse", "not in corpus", "no evidence", "unable")
        )
    expected = str(row.get("answer_text") or row.get("expected_answer") or "").strip()
    if not expected:
        return False
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", expected)
    if numbers:
        return all(number.replace(",", "") in result.answer.replace(",", "") for number in numbers)
    return expected.lower() in result.answer.lower()


def _hit_at_k(row, result):
    document_id, page = _gold_evidence(row)
    if not document_id:
        return None
    for entry in (result.state or {}).get("tool_results", []):
        if entry.get("tool") != "search_index":
            continue
        for hit in ((entry.get("result") or {}).get("hits") or []):
            if not str(hit.get("chunk_id") or "").startswith(str(document_id)):
                continue
            if page is None or (hit.get("meta") or {}).get("page") == page:
                return True
    return False


def _build_record(row, label, result, latency):
    tools = [
        entry.get("name")
        for entry in (getattr(result, "events", None) or [])
        if entry.get("type") == "tool" or entry.get("event") == "tool"
    ]
    citations = list(getattr(result, "citations", None) or [])
    return {
        "row_id": row.get("row_id") or row.get("id"),
        "config": label,
        "question": row.get("question") or row.get("question_text"),
        "expected_behaviour": row.get("expected_behaviour"),
        "correct": bool(_is_correct(row, result)),
        "hit_at_k": _hit_at_k(row, result),
        "tools": tools,
        "latency_s": round(float(latency), 3),
        "citation_present": bool(citations)
        and all(re.search(r"\[[0-9a-f]{12} p\.\d+\]", citation) for citation in citations),
        "stop_reason": getattr(result, "stop_reason", ""),
        "answer": getattr(result, "answer", ""),
        "citations": citations,
    }


def _error_record(row, label, exc, latency):
    return {
        "row_id": row.get("row_id") or row.get("id"),
        "config": label,
        "question": row.get("question") or row.get("question_text"),
        "expected_behaviour": row.get("expected_behaviour"),
        "correct": False,
        "hit_at_k": None,
        "tools": [],
        "latency_s": round(float(latency), 3),
        "citation_present": False,
        "stop_reason": f"error: {type(exc).__name__}: {exc}",
        "answer": "",
        "citations": [],
    }


def _usable(path):
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(record.get("stop_reason") or "").startswith("error:") is False


def _write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in record.items()
            })


def _summarize(records):
    count = len(records)
    hits = [record["hit_at_k"] for record in records if record["hit_at_k"] is not None]
    return {
        "n": count,
        "accuracy": round(sum(record["correct"] for record in records) / count, 3) if count else 0.0,
        "recall_at_k": round(sum(bool(hit) for hit in hits) / len(hits), 3) if hits else 0.0,
        "citation_rate": round(sum(record["citation_present"] for record in records) / count, 3) if count else 0.0,
        "avg_latency_s": round(sum(record["latency_s"] for record in records) / count, 3) if count else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/hybrid_eval_v2")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--configs", default="OFF,ON")
    args = parser.parse_args()

    from src.agent.service import run_question
    from src.agent.tools import warm_query_embedding

    os.environ["AGENT_RUN_TIMEOUT"] = "600"
    rows = [
        json.loads(line)
        for line in (ROOT / "data/eval/datasets/gold_v2_eval_v8_public.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    out_dir = args.out
    query_dir = out_dir / "per_query"
    out_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    warm_started = time.monotonic()
    warm_latency = warm_query_embedding(timeout_s=600.0)
    print(f"warmup_latency_s={warm_latency:.3f} wall_s={time.monotonic() - warm_started:.3f}", flush=True)

    for label in [item.strip().upper() for item in args.configs.split(",") if item.strip()]:
        enabled = "1" if label == "ON" else "0"
        os.environ["RETRIEVAL_HYBRID"] = enabled
        os.environ["RETRIEVAL_RERANK"] = enabled
        print(f"=== RUN {label} ===", flush=True)
        for index, row in enumerate(rows, 1):
            row_id = row.get("row_id") or row.get("id") or f"row-{index}"
            path = query_dir / f"{label.lower()}_{row_id}.json"
            if args.resume and _usable(path):
                print(f"[{label}] {index}/{len(rows)} {row_id} resume", flush=True)
                continue
            started = time.monotonic()
            try:
                result = run_question(
                    row.get("question") or row.get("question_text"),
                    session_id=f"hybrid-eval-v2-{enabled}-{row_id}",
                )
                latency = getattr(result, "latency_s", None) or time.monotonic() - started
                record = _build_record(row, label, result, latency)
            except Exception as exc:
                record = _error_record(row, label, exc, time.monotonic() - started)
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(
                f"[{label}] {index}/{len(rows)} {row_id} correct={record['correct']} "
                f"hit={record['hit_at_k']} cite={record['citation_present']} "
                f"tools={record['tools']} latency={record['latency_s']}",
                flush=True,
            )
            time.sleep(1.5)

    summary = {}
    for label in [item.strip().upper() for item in args.configs.split(",") if item.strip()]:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(query_dir.glob(f"{label.lower()}_*.json"))
        ]
        records.sort(key=lambda item: rows.index(next(row for row in rows if (row.get("row_id") or row.get("id")) == item["row_id"])))
        (out_dir / f"hybrid_eval_{label.lower()}.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _write_csv(out_dir / f"hybrid_eval_{label.lower()}.csv", records)
        summary[label] = _summarize(records)
        summary[label]["cold_start_query_s"] = records[0]["latency_s"] if records else 0.0
    summary["warmup_latency_s"] = round(warm_latency, 3)
    (out_dir / "hybrid_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
