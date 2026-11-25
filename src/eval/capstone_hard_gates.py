from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import GOLD_PATH
from src.eval import agent_eval
from src.eval.prompting.context import candidate_pages
from src.eval.prompting.scoring import score_prediction
from src.eval.prompting.strategies import (
    build_s1_messages,
    build_s2_messages,
    build_s3_step1_messages,
)
from src.eval.prompts import render_prompt
from src.eval.retrieval import hit_at_k

RUN_ID = "capstone_hard_gates_v1"
OUT_DIR = Path("results/evaluation") / RUN_ID
GOLD = Path(GOLD_PATH)
FREEZE = Path("data/corpus/freeze_v1.json")
KS = (1, 3, 5, 10)
ARMS = ("S1", "S2", "S3", "AGENT")
SOURCE = "offline_oracle_standin; strategy prompts and gold evidence reused; no live model calls"
PRICE_SOURCE = "college endpoint: no published price; cost proxy is tokens only"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gold_page(row: dict[str, Any]) -> int | None:
    for evidence in row.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        for key in ("pdf_page", "pdf_page_index", "page"):
            value = evidence.get(key)
            if isinstance(value, int):
                return value
    value = row.get("pdf_page_index")
    return value if isinstance(value, int) else None


def _gold_doc(row: dict[str, Any]) -> str:
    for evidence in row.get("evidence") or []:
        if isinstance(evidence, dict) and evidence.get("document_id"):
            return str(evidence["document_id"])
    return str(row.get("document_id") or "0" * 12)


def _expected_answer(row: dict[str, Any]) -> str:
    return str(row.get("expected_answer") or row.get("answer_text") or "").strip()


def _context_pairs(row: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (p["document_id"], p["page"])
        for p in candidate_pages(row)
        if isinstance(p.get("document_id"), str) and isinstance(p.get("page"), int)
    }


def _rendered_prompt_hashes(arm: str, row: dict[str, Any]) -> list[str]:
    if arm == "S1":
        messages = build_s1_messages(row, context="", few_shot_rows=[])
        prompts = [render_prompt(messages)]
    elif arm == "S2":
        messages = build_s2_messages(row, context="")
        prompts = [render_prompt(messages)] * 3
    else:
        prompts = [
            render_prompt(build_s3_step1_messages(row, context="")),
            "step2: calculate from extracted values",
            "step3: answer from extracted and computed values",
        ]
    return [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in prompts]


def _offline_output(arm: str, row: dict[str, Any]) -> str:
    if row.get("expected_behaviour") == "abstain":
        return "ABSTAIN"
    expected = _expected_answer(row)
    if not expected:
        return "ABSTAIN"
    page = _gold_page(row)
    if page is None:
        return "ABSTAIN"
    if arm == "S1":
        payload = {
            "answer_value": expected,
            "answer_kind": "numeric",
            "page": page,
            "citations": [{"document_id": _gold_doc(row), "page": page}],
        }
        return json.dumps(payload, ensure_ascii=False)
    if arm == "S2":
        return f"answer: {expected} | page: {page}"
    return f"decomposition: locate, extract, verify; answer: {expected} | page: {page}"


def _prediction_correct(row: dict[str, Any], score: dict[str, Any]) -> bool:
    if row.get("expected_behaviour") == "abstain":
        return bool(score["abstain_correct"])
    kind = score.get("gold_answer_kind")
    if kind == "numeric" and score.get("numeric_match") is not None:
        return bool(score["numeric_match"])
    return bool(score.get("entity_match"))


def _score_baseline(arm: str, row: dict[str, Any]) -> dict[str, Any]:
    row_copy = dict(row)
    pairs = _context_pairs(row_copy)
    output = _offline_output(arm, row_copy)
    score = score_prediction(row_copy, output, pairs)
    return {
        "row_id": row.get("row_id") or row.get("id"),
        "arm": arm,
        "source": SOURCE,
        "status": "offline_oracle_standin",
        "output": output,
        "correct": _prediction_correct(row_copy, score),
        "abstain_correct": bool(score["abstain_correct"]),
        "citation_valid": score["citation_valid"],
        "parse_ok": bool(score["parse_ok"]),
        "prompt_hashes": _rendered_prompt_hashes(arm, row_copy),
        "tokens": 0,
        "latency_ms": 0,
        "tool_calls": 0,
        "score": score,
    }


def _score_agent(row: dict[str, Any]) -> dict[str, Any]:
    output = agent_eval.mock_agent(row.get("question_text") or row.get("question") or "", row=row)
    scored = agent_eval.score_row(row, output)
    scored["source"] = SOURCE
    return scored


def _retrieval_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = str(row.get("row_id") or row.get("id"))
        pairs: set[tuple[str, int]] = set()
        for item in row.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            doc = item.get("document_id")
            page = item.get("pdf_page")
            if doc and isinstance(page, int):
                pairs.add((str(doc)[:12], page))
        retrieved = [
            {"meta": {"document_id": str(p["document_id"])[:12], "page": p["page"]}}
            for p in candidate_pages(row)
        ]
        per_row = {"row_id": rid, "eligible": bool(pairs), "gold_pairs": sorted(pairs)}
        for k in KS:
            per_row[f"hit@{k}"] = hit_at_k(retrieved, pairs, k) if pairs else None
        by_id[rid] = per_row
    eligible = [item for item in by_id.values() if item["eligible"]]
    hits = {f"recall@{k}": sum(1 for item in eligible if item[f"hit@{k}"]) for k in KS}
    report = {
        "eligible_rows": len(eligible),
        "excluded_rows_without_gold_evidence": len(by_id) - len(eligible),
        "hits": hits,
        "by_row": by_id,
    }
    for k in KS:
        denominator = report["eligible_rows"] or 1
        report[f"recall@{k}"] = round(hits[f"recall@{k}"] / denominator, 4)
    return report, by_id


def _accuracy(value: bool | None, denominator: int) -> float | None:
    if denominator == 0 or value is None:
        return None
    return round(float(value) / denominator, 4)


def _baseline_summary(
    arm: str, rows: list[dict[str, Any]], retrieval: dict[str, Any]
) -> dict[str, Any]:
    n = len(rows)
    answer_rows = [r for r in rows if r["score"]["gold_abstain"] is False]
    abstain_rows = [r for r in rows if r["score"]["gold_abstain"] is True]
    correct_answer = sum(1 for r in answer_rows if r["correct"])
    correct_abstain = sum(1 for r in abstain_rows if r["correct"])
    overall = correct_answer + correct_abstain
    return {
        "arm": arm,
        "source": SOURCE,
        "n": n,
        "answer_rows": len(answer_rows),
        "abstain_rows": len(abstain_rows),
        "accuracy": _accuracy(overall, n),
        "answer_accuracy": _accuracy(correct_answer, len(answer_rows)),
        "abstain_accuracy": _accuracy(correct_abstain, len(abstain_rows)),
        "citation_validity": _accuracy(
            sum(1 for r in answer_rows if r["citation_valid"] is True), len(answer_rows)
        ),
        "task_completion": None,
        "avg_tool_calls": 0.0,
        "cost_per_query": 0.0,
        "latency_ms_per_query": 0.0,
        "recall_at_5": retrieval.get("recall@5"),
    }


def _agent_summary(scores: list[dict[str, Any]], retrieval: dict[str, Any]) -> dict[str, Any]:
    metrics = agent_eval.aggregate(scores)
    n = len(scores)
    answer_rows = [r for r in scores if r["expected_behaviour"] == "answer"]
    abstain_rows = [r for r in scores if r["expected_behaviour"] == "abstain"]
    correct = sum(1 for r in answer_rows if r["answer_correct"])
    refused = sum(1 for r in abstain_rows if r["refusal_correct"])
    return {
        "arm": "AGENT",
        "source": SOURCE,
        "n": n,
        "answer_rows": len(answer_rows),
        "abstain_rows": len(abstain_rows),
        "accuracy": _accuracy(correct + refused, n),
        "answer_accuracy": _accuracy(correct, len(answer_rows)),
        "abstain_accuracy": _accuracy(refused, len(abstain_rows)),
        "citation_validity": metrics.get("citation_correctness"),
        "task_completion": metrics.get("task_completion_rate"),
        "avg_tool_calls": metrics.get("mean_tool_calls"),
        "cost_per_query": metrics.get("mean_cost_proxy_usd"),
        "latency_ms_per_query": metrics.get("mean_latency_ms"),
        "recall_at_5": retrieval.get("recall@5"),
        "metrics": metrics,
    }


def _paired_counts(
    agent_scores: list[dict[str, Any]], baseline_scores: list[dict[str, Any]]
) -> dict[str, int]:
    agent_by_id = {str(r["row_id"]): r for r in agent_scores}
    counts = Counter()
    for row in baseline_scores:
        agent = agent_by_id.get(str(row["row_id"]))
        if agent is None:
            continue
        a = bool(
            agent["answer_correct"]
            if agent["expected_behaviour"] == "answer"
            else agent["refusal_correct"]
        )
        b = bool(row["correct"])
        key = (
            "both_passed"
            if a and b
            else "agent_only"
            if a
            else "baseline_only"
            if b
            else "both_failed"
        )
        counts[key] += 1
    return {
        name: counts.get(name, 0)
        for name in ("both_passed", "both_failed", "agent_only", "baseline_only")
    }


def _failure_taxonomy(
    agent_scores: list[dict[str, Any]], retrieval_by_row: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for score in agent_scores:
        rid = str(score["row_id"])
        failures = set(score.get("failures") or [])
        labels: list[str] = []
        gold_answer = score.get("expected_behaviour") == "answer"
        if gold_answer and retrieval_by_row.get(rid, {}).get("hit@10") is False:
            labels.append("retrieval_miss")
        if "tool_failure" in failures:
            labels.append("tool_call_error")
        if "incorrect_answer" in failures and "citation_issue" in failures:
            labels.append("hallucination")
        if {
            "incorrect_answer",
            "incomplete_task",
            "loop",
            "token_exhaustion",
            "timeout",
        } & failures:
            labels.append("reasoning_error")
        for label in labels:
            counts[label] += 1
        if labels:
            rows.append({"row_id": rid, "labels": labels})
    n = len(agent_scores) or 1
    return {
        "denominator": len(agent_scores),
        "counts": {
            name: counts.get(name, 0)
            for name in ("retrieval_miss", "reasoning_error", "tool_call_error", "hallucination")
        },
        "rates": {
            name: round(counts.get(name, 0) / n, 4)
            for name in ("retrieval_miss", "reasoning_error", "tool_call_error", "hallucination")
        },
        "rows": rows,
    }


def _reverse_index(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    index: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        rid = str(row.get("row_id") or row.get("id"))
        for item in row.get("evidence") or []:
            if isinstance(item, dict) and item.get("document_id"):
                key = str(item["document_id"])
                if rid not in index[key]:
                    index[key].append(rid)
        doc = row.get("document_id")
        if doc and rid not in index[str(doc)]:
            index[str(doc)].append(rid)
    return {key: sorted(value) for key, value in sorted(index.items())}


def _report_lint(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    comparison_fields = {
        "arm",
        "source",
        "n",
        "accuracy",
        "task_completion",
        "avg_tool_calls",
        "cost_per_query",
        "latency_ms_per_query",
        "recall_at_5",
        "both_passed",
        "both_failed",
        "agent_only",
        "baseline_only",
    }
    for row in payload["comparison"]:
        required = set(comparison_fields)
        if row.get("arm") == "AGENT":
            required -= {"both_passed", "both_failed", "agent_only", "baseline_only"}
        missing = required - set(row)
        if missing:
            errors.append(f"{row.get('arm')}: missing tuple fields {sorted(missing)}")
    for key, value in payload["metrics"].items():
        if isinstance(value, float) and ("accuracy" in key or key.endswith("_rate")):
            if "denominator" not in payload["metrics"]:
                errors.append(f"{key}: percentage without denominator")
    return {"ok": not errors, "errors": errors}


def _markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    retrieval = payload["retrieval"]
    taxonomy = payload["failure_taxonomy"]
    lines = [
        "# Capstone hard-gates evaluation summary",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Gold: `{payload['gold_path']}`; sha256 `{payload['gold_sha256']}`",
        f"- Split: `{payload['split']}`; rows `{metrics['n']}`; "
        f"holdout rows excluded `{payload['holdout_rows_excluded']}`",
        f"- Corpus freeze: `{payload['corpus_freeze_file_sha256']}`",
        f"- Cost proxy: `{payload['price_source']}`",
        "- Caveat: offline oracle stand-in validates the harness and report contracts; "
        "it is not live model accuracy.",
        "",
        "## Summary metrics",
        "",
        "| Metric | Value | Denominator |",
        "| --- | ---: | ---: |",
    ]
    metric_rows = [
        ("End-to-end accuracy", metrics["overall_accuracy"], metrics["n"]),
        ("Answer accuracy", metrics["answer_accuracy"], metrics["n_answer"]),
        ("Abstain accuracy", metrics["abstain_accuracy"], metrics["n_abstain"]),
        ("Task completion rate", metrics["task_completion_rate"], metrics["n"]),
        ("Avg tool calls/query", metrics["mean_tool_calls"], metrics["n"]),
        ("Cost proxy/query USD", metrics["mean_cost_proxy_usd"], metrics["n"]),
        ("Latency ms/query", metrics["mean_latency_ms"], metrics["n"]),
        ("Mean tokens/query", metrics["mean_tokens"], metrics["n"]),
    ]
    for name, value, denominator in metric_rows:
        lines.append(f"| {name} | {value} | {denominator} |")
    lines += [
        "",
        "## Retrieval recall@k (reported separately)",
        "",
        "| Metric | Hits | Denominator | Value |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in KS:
        key = f"recall@{k}"
        lines.append(
            f"| recall@{k} | {retrieval['hits'][key]} | "
            f"{retrieval['eligible_rows']} | {retrieval[key]} |"
        )
    lines += [
        "",
        "## Failure taxonomy",
        "",
        "| Label | Count | Denominator | Rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, count in taxonomy["counts"].items():
        lines.append(
            f"| {name} | {count} | {taxonomy['denominator']} | {taxonomy['rates'][name]} |"
        )
    lines += [
        "",
        "## Baseline vs agent (same run)",
        "",
        "| Arm | Source | n | Accuracy | Completion | Tool calls | "
        "Cost/query | Latency ms/query | Recall@5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["comparison"]:
        completion = "n/a" if row["task_completion"] is None else row["task_completion"]
        lines.append(
            f"| {row['arm']} | `{row['source']}` | {row['n']} | {row['accuracy']} | "
            f"{completion} | {row['avg_tool_calls']} | {row['cost_per_query']} | "
            f"{row['latency_ms_per_query']} | {row['recall_at_5']} |"
        )
    lines += [
        "",
        "## Paired comparison counts",
        "",
        "| Arm | Both passed | Both failed | Agent only | Baseline only |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["comparison"]:
        if row["arm"] == "AGENT":
            continue
        lines.append(
            f"| {row['arm']} vs AGENT | {row['both_passed']} | {row['both_failed']} | "
            f"{row['agent_only']} | {row['baseline_only']} |"
        )
    lines += [
        "",
        f"Report lint: `{payload['report_lint']['ok']}`; "
        f"errors `{payload['report_lint']['errors']}`",
        f"Document-to-questions index documents: `{len(payload['document_to_questions'])}`",
        "",
    ]
    return "\n".join(lines)


def build_payload() -> dict[str, Any]:
    started = time.monotonic()
    gold_rows = agent_eval.load_gold(GOLD)
    rows = [r for r in gold_rows if r.get("split") == "dev"]
    if not rows:
        raise SystemExit(f"no dev rows in {GOLD}")
    retrieval, retrieval_by_row = _retrieval_rows(rows)
    baseline_scores = {arm: [_score_baseline(arm, r) for r in rows] for arm in ("S1", "S2", "S3")}
    agent_scores = [_score_agent(r) for r in rows]
    agent_metrics = agent_eval.aggregate(agent_scores)
    answer_rows = [r for r in agent_scores if r["expected_behaviour"] == "answer"]
    abstain_rows = [r for r in agent_scores if r["expected_behaviour"] == "abstain"]
    metrics = {
        "n": len(rows),
        "n_answer": len(answer_rows),
        "n_abstain": len(abstain_rows),
        "overall_accuracy": _accuracy(
            sum(1 for r in answer_rows if r["answer_correct"])
            + sum(1 for r in abstain_rows if r["refusal_correct"]),
            len(rows),
        ),
        "denominator": len(rows),
        **agent_metrics,
    }
    metrics["abstain_accuracy"] = agent_metrics.get("refusal_correctness")
    taxonomy = _failure_taxonomy(agent_scores, retrieval_by_row)
    comparison = [
        _baseline_summary(arm, baseline_scores[arm], retrieval) for arm in ("S1", "S2", "S3")
    ]
    agent_summary = _agent_summary(agent_scores, retrieval)
    for row in comparison:
        row.update(_paired_counts(agent_scores, baseline_scores[row["arm"]]))
    comparison.append(agent_summary)
    payload = {
        "run_id": RUN_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": SOURCE,
        "gold_path": str(GOLD),
        "gold_sha256": _sha256(GOLD),
        "corpus_freeze_file_sha256": _sha256(FREEZE),
        "corpus_freeze_manifest_sha256": json.loads(FREEZE.read_text(encoding="utf-8"))[
            "manifest_sha256"
        ],
        "split": "dev",
        "holdout_rows_excluded": sum(1 for r in gold_rows if r.get("split") == "holdout"),
        "source": SOURCE,
        "price_source": PRICE_SOURCE,
        "prompt_strategies": [
            "S1 structured few-shot",
            "S2 CoT self-consistency",
            "S3 decomposition",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "metrics": metrics,
        "retrieval": retrieval,
        "failure_taxonomy": taxonomy,
        "comparison": comparison,
        "document_to_questions": _reverse_index(rows),
        "per_row": {
            "agent": agent_scores,
            "S1": baseline_scores["S1"],
            "S2": baseline_scores["S2"],
            "S3": baseline_scores["S3"],
        },
    }
    payload["report_lint"] = _report_lint(payload)
    return payload


def main() -> int:
    payload = build_payload()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_json = OUT_DIR / "results.json"
    summary_md = OUT_DIR / "summary.md"
    results_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary_md.write_text(_markdown(payload), encoding="utf-8")
    print(f"results_json={results_json}")
    print(f"summary_md={summary_md}")
    print(
        _markdown(payload)
        .split("## Baseline vs agent (same run)", 1)[1]
        .split("## Paired comparison counts", 1)[0]
        .strip()
    )
    return 0 if payload["report_lint"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
