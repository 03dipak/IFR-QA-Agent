"""Append-only, resumable runner for Week-1/Week-2 prompt arms."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from src.eval.gold_rows import assert_no_holdout
from src.eval.prompting.context import candidate_pages, pack_pages, rag_candidate_pages
from src.eval.prompting.fewshot import load_few_shot as load_prompt_few_shot
from src.eval.prompting.inputs import load_dev_rows
from src.eval.prompting.params import model_params, model_params_hash
from src.eval.prompting.scoring import normalize_answer_value, score_prediction
from src.eval.prompting.strategies import (
    build_s1_messages, build_s2_messages, build_s3_step1_messages,
    build_s3_step2_messages, build_s3_step3_messages,
)
from src.eval.contexts import load_few_shot as load_dev_few_shot
from src.eval.prompts import build_messages, prompt_set_hash, render_prompt
from src.config import GOLD_PATH
from src.llm.client import LLMError, chat

GOLD = Path(GOLD_PATH)
FREEZE = Path("data/corpus/freeze_v1.json")
OUT_DIR = Path("results/week1_prompt_baseline/runs")


def _gold_hash() -> str:
    return hashlib.sha256(GOLD.read_bytes()).hexdigest()


def _corpus_freeze_sha() -> str:
    return json.loads(FREEZE.read_text(encoding="utf-8"))["manifest_sha256"]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None

_HOLDOUT_IDS: set[str] | None = None


def _holdout_ids() -> set[str]:
    global _HOLDOUT_IDS
    if _HOLDOUT_IDS is None:
        from src.eval.prompting.inputs import load_holdout_ids
        _HOLDOUT_IDS = load_holdout_ids()
    return _HOLDOUT_IDS


def _assert_no_holdout_in_prompts(prompts: list[str], row_id: str | None = None) -> None:
    ids = _holdout_ids()
    for pr in prompts:
        for hid in ids:
            if hid and hid in (pr or ""):
                raise AssertionError(f"holdout id {hid} present in prompt for {row_id}")


def _load_completed(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if rec.get("status") in {"answered", "BUDGET_EXCEEDED", "FILTERED_EMPTY"}:
                done.add(rec["row_id"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"corrupt checkpoint line in {path}: {line[:120]}") from exc
    return done


def _arm_params(arm: str) -> tuple[float, int]:
    if arm == "S2":
        return 0.7, 1024
    return 0.0, 512


def _rag_filtered(arm: str, retrieval_variant: str | None) -> bool:
    if retrieval_variant is not None:
        return retrieval_variant == "filtered"
    return arm.endswith("filtered")


def _context_for(arm: str, row: dict, *, max_tokens: int,
                 retrieval_variant: str | None = None) -> tuple[str | None, set[tuple[str, int]], bool, list[dict], str | None]:
    if arm == "B0":
        return None, set(), True, [], None
    if arm.startswith("B2"):
        filtered = _rag_filtered(arm, retrieval_variant)
        pages = rag_candidate_pages(row, filtered=filtered)
        if filtered and not pages:
            return None, set(), False, [], "FILTERED_EMPTY"
    else:
        pages = candidate_pages(row)
    system = "You are a precise financial-results answerer."
    question = row.get("question_text") or row.get("question") or ""
    context, keys, ok = pack_pages(pages, system, question, max_tokens, 8192)
    return context, keys, ok, pages, (None if ok else "BUDGET_EXCEEDED")


def rendered_prompts_for(arm: str, row: dict, *, few_shot: list[dict] | None = None,
                          retrieval_variant: str | None = None) -> list[str]:
    max_tokens = 1024 if arm == "S2" else 512
    context, _keys, packed, _pages, _status = _context_for(arm, row, max_tokens=max_tokens,
                                                     retrieval_variant=retrieval_variant)
    if arm != "B0" and not packed and _status != "FILTERED_EMPTY" and (
            (arm.startswith("B2") and True) or (not arm.startswith("B2") and bool(candidate_pages(row)))):
        context = None
    if arm == "B0":
        messages = build_messages("B0", row, context=None, few_shot_rows=few_shot)
        return [render_prompt(messages)]
    if arm == "B1":
        messages = build_messages("B1", row, context=context)
        return [render_prompt(messages)]
    if arm == "S1":
        row_doc_ids = {did for did in ([row.get("document_id")] + [e.get("document_id") for e in (row.get("evidence") or []) if isinstance(e, dict)]) if isinstance(did, str) and did}
        fs = few_shot if few_shot is not None else load_prompt_few_shot(exclude_doc_ids=row_doc_ids)
        messages = build_s1_messages(row, context=context, few_shot_rows=fs)
        return [render_prompt(messages)]
    if arm == "S2":
        messages = build_s2_messages(row, context=context)
        return [render_prompt(messages)] * 3
    if arm == "S3":
        step1 = build_s3_step1_messages(row, context=context)
        step2 = build_s3_step2_messages("[]")
        step3 = build_s3_step3_messages(row, extracted="[]", computed="{}")
        return [render_prompt(step1), render_prompt(step2), render_prompt(step3)]
    if arm.startswith("B2"):
        rv = "filtered" if (retrieval_variant == "filtered" or arm.endswith("filtered")) else "global"
        messages = build_messages("B2", row, context=context, retrieval_variant=rv)
        return [render_prompt(messages)]
    raise ValueError(f"unknown arm: {arm}")


def run_arm(arm: str, *, rows: list[dict] | None = None, smoke: bool = False,
            live: bool = True, limit: int | None = None,
            run_id: str | None = None, few_shot: list[dict] | None = None,
            retrieval_variant: str | None = None, out_dir: Path | str | None = None) -> dict:
    if rows is None:
        rows = load_dev_rows()
    assert_no_holdout(rows, context=f"week2 {arm} run")
    if arm == "S2":
        rows = [r for r in rows if r.get("expected_behaviour") == "answer"]
    if limit is not None:
        rows = rows[:limit]
    assert_no_holdout(rows, context=f"week2 {arm} run (limited)")

    run_id = run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{arm}"
    od = Path(out_dir) if out_dir else OUT_DIR
    od.mkdir(parents=True, exist_ok=True)
    out_path = od / f"{run_id}_{arm}.jsonl"
    manifest_path = od / f"{run_id}_{arm}.manifest.json"
    completed = _load_completed(out_path)

    temperature, max_tokens = _arm_params(arm)
    params = model_params(temperature=temperature, max_tokens=max_tokens)
    params_hash = model_params_hash(temperature=temperature, max_tokens=max_tokens)

    manifest = {
        "run_id": run_id, "arm": arm, "split": "dev", "smoke": bool(smoke),
        "label": "SMOKE_NOT_SCORED" if smoke else "SCORED",
        "started_at": datetime.now(UTC).isoformat(),
        "prompt_set_version": __import__("src.eval.prompts", fromlist=["PROMPT_SET_VERSION"]).PROMPT_SET_VERSION,
        "prompt_set_hash": prompt_set_hash(),
        "model_params": params,
        "model_params_hash": params_hash,
        "gold_hash": _gold_hash(),
        "corpus_freeze_manifest_sha256": _corpus_freeze_sha(),
        "git_commit": _git_commit(),
        "n_rows": len(rows),
        "n_completed_before": len(completed),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    processed = 0
    errors = 0
    stopped_reason = None

    for row in rows:
        rid = row.get("id") or row.get("row_id")
        if rid in completed:
            continue

        context, context_keys, packed, _pages, budget_status = _context_for(arm, row, max_tokens=max_tokens,
                                                    retrieval_variant=retrieval_variant)

        if not packed and budget_status == "FILTERED_EMPTY" and arm.startswith("B2") and _rag_filtered(arm, retrieval_variant):
            messages = []
            rendered = ""
            rendered_hashes = []
            status = "FILTERED_EMPTY"
            output = "ABSTAIN"
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            http_status = None
            latency_ms = 0
            context = None
        elif not packed:
            messages = []
            rendered = ""
            rendered_hashes = []
            status = "BUDGET_EXCEEDED"
            output = "BUDGET_EXCEEDED: prompt/context exceeds 8192-token budget"
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            http_status = None
            latency_ms = 0
            context = None
        elif arm == "S2":
            fs = None
            msgs = build_s2_messages(row, context=context)
            rendered = render_prompt(msgs)
            _assert_no_holdout_in_prompts([rendered], rid)
            rendered_hashes = [hashlib.sha256(rendered.encode("utf-8")).hexdigest() for _ in range(3)]
            samples = []
            http_status = None
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            latency_ms = 0
            status = "answered"
            if not live:
                output = ""
                status = "not_live"
                for _ in range(3):
                    samples.append({"status": "not_live", "output": "", "usage": {}, "latency_ms": 0})
            else:
                for _ in range(3):
                    t0 = time.time()
                    try:
                        resp = chat(msgs, **params)
                        sample_output = resp.text
                        sample_usage = resp.usage or {}
                        sample_status = "answered" if resp.status == 200 else "error"
                        sample_http = resp.status
                    except LLMError as exc:
                        sample_output = ""
                        sample_usage = {}
                        sample_status = "error"
                        sample_http = None
                        _ = str(exc)
                    sample_latency = int((time.time() - t0) * 1000)
                    samples.append({"status": sample_status, "http_status": sample_http,
                                    "output": sample_output, "usage": sample_usage,
                                    "latency_ms": sample_latency})
                    usage["prompt_tokens"] += sample_usage.get("prompt_tokens", 0) or 0
                    usage["completion_tokens"] += sample_usage.get("completion_tokens", 0) or 0
                    usage["total_tokens"] += sample_usage.get("total_tokens", 0) or 0
                    latency_ms += sample_latency
                    if sample_status == "error":
                        status = "error"
                if status != "error":
                    status = "answered"
                if samples:
                    http_status = samples[0].get("http_status")
            if status == "answered":
                norms = [normalize_answer_value(s["output"]) for s in samples]
                counts = {}
                for norm in norms:
                    counts[norm] = counts.get(norm, 0) + 1
                best_norm = max(norms, key=lambda n: (counts[n], -norms.index(n)))
                selected = norms.index(best_norm)
                output = samples[selected]["output"]
            else:
                output = ""
            messages = msgs
            rendered = render_prompt(messages)
            rendered_hashes = [hashlib.sha256(rendered.encode("utf-8")).hexdigest() for _ in range(3)]
            context_pages = sorted([list(k) for k in context_keys])
            record_extra = {"samples": samples, "majority_samples": 3}
        elif arm == "S3":
            fs = None
            step_messages = build_s3_step1_messages(row, context=context)
            rendered = render_prompt(step_messages)
            _assert_no_holdout_in_prompts([rendered], rid)
            rendered_hashes = [hashlib.sha256(rendered.encode("utf-8")).hexdigest()]
            steps = []
            http_status = None
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            latency_ms = 0
            status = "answered"
            output = ""
            extracted = ""
            computed = ""
            if not live:
                steps.append({"step": 1, "status": "not_live", "output": "", "usage": {}, "latency_ms": 0})
                status = "not_live"
            else:
                t0 = time.time()
                try:
                    resp = chat(step_messages, **model_params(temperature=0.0, max_tokens=512))
                    extracted = resp.text
                    step_usage = resp.usage or {}
                    step_status = "answered" if resp.status == 200 else "error"
                    step_http = resp.status
                except LLMError as exc:
                    extracted = ""
                    step_usage = {}
                    step_status = "error"
                    step_http = None
                    _ = str(exc)
                step_latency = int((time.time() - t0) * 1000)
                steps.append({"step": 1, "status": step_status, "http_status": step_http,
                              "output": extracted, "usage": step_usage, "latency_ms": step_latency,
                              "rendered_prompt_hash": rendered_hashes[0]})
                usage["prompt_tokens"] += step_usage.get("prompt_tokens", 0) or 0
                usage["completion_tokens"] += step_usage.get("completion_tokens", 0) or 0
                usage["total_tokens"] += step_usage.get("total_tokens", 0) or 0
                latency_ms += step_latency
                if step_status != "answered":
                    status = "error" if step_status == "error" else step_status
                if status == "answered":
                    step2_messages = build_s3_step2_messages(extracted)
                    rendered2 = render_prompt(step2_messages)
                    rendered_hashes.append(hashlib.sha256(rendered2.encode("utf-8")).hexdigest())
                    t0 = time.time()
                    try:
                        resp = chat(step2_messages, **model_params(temperature=0.0, max_tokens=512))
                        computed = resp.text
                        step_usage = resp.usage or {}
                        step_status = "answered" if resp.status == 200 else "error"
                        step_http = resp.status
                    except LLMError as exc:
                        computed = ""
                        step_usage = {}
                        step_status = "error"
                        step_http = None
                        _ = str(exc)
                    step_latency = int((time.time() - t0) * 1000)
                    steps.append({"step": 2, "status": step_status, "http_status": step_http,
                                  "output": computed, "usage": step_usage, "latency_ms": step_latency,
                                  "rendered_prompt_hash": rendered_hashes[-1]})
                    usage["prompt_tokens"] += step_usage.get("prompt_tokens", 0) or 0
                    usage["completion_tokens"] += step_usage.get("completion_tokens", 0) or 0
                    usage["total_tokens"] += step_usage.get("total_tokens", 0) or 0
                    latency_ms += step_latency
                    if step_status != "answered":
                        status = "error" if step_status == "error" else step_status
                if status == "answered":
                    step3_messages = build_s3_step3_messages(row, extracted=extracted, computed=computed)
                    rendered3 = render_prompt(step3_messages)
                    rendered_hashes.append(hashlib.sha256(rendered3.encode("utf-8")).hexdigest())
                    t0 = time.time()
                    try:
                        resp = chat(step3_messages, **model_params(temperature=0.0, max_tokens=512))
                        output = resp.text
                        step_usage = resp.usage or {}
                        step_status = "answered" if resp.status == 200 else "error"
                        step_http = resp.status
                    except LLMError as exc:
                        output = ""
                        step_usage = {}
                        step_status = "error"
                        step_http = None
                        _ = str(exc)
                    step_latency = int((time.time() - t0) * 1000)
                    steps.append({"step": 3, "status": step_status, "http_status": step_http,
                                  "output": output, "usage": step_usage, "latency_ms": step_latency,
                                  "rendered_prompt_hash": rendered_hashes[-1]})
                    usage["prompt_tokens"] += step_usage.get("prompt_tokens", 0) or 0
                    usage["completion_tokens"] += step_usage.get("completion_tokens", 0) or 0
                    usage["total_tokens"] += step_usage.get("total_tokens", 0) or 0
                    latency_ms += step_latency
                    if step_status != "answered":
                        status = "error" if step_status == "error" else step_status
            if status not in {"error", "BUDGET_EXCEEDED", "FILTERED_EMPTY"} and not live:
                status = "not_live"
            if steps:
                http_status = steps[0].get("http_status")
            context_pages = sorted([list(k) for k in context_keys])
            record_extra = {"steps": steps}
        else:
            few_shot = few_shot
            row_doc_ids = set()
            did = row.get("document_id")
            if isinstance(did, str) and did:
                row_doc_ids.add(did)
            for ev in row.get("evidence") or []:
                ed = ev.get("document_id") if isinstance(ev, dict) else None
                if isinstance(ed, str) and ed:
                    row_doc_ids.add(ed)
            if arm == "B0" and few_shot is None:
                few_shot = load_dev_few_shot(n=3, exclude_id=rid, exclude_doc_ids=row_doc_ids)
            if arm == "S1" and few_shot is None:
                few_shot = load_prompt_few_shot(exclude_doc_ids=row_doc_ids)
            rv = "filtered" if (retrieval_variant == "filtered" or (retrieval_variant is None and arm.endswith("filtered"))) else "global"
            messages = build_messages((arm.split("-", 1)[0] if arm.startswith("B2") else arm),
                                      row, context=context,
                                      few_shot_rows=few_shot if arm in {"B0", "S1"} else None,
                                      retrieval_variant=rv)
            if arm == "S1":
                from src.eval.prompting.strategies import build_s1_messages
                messages = build_s1_messages(row, context=context, few_shot_rows=few_shot or [])
            rendered = render_prompt(messages)
            _assert_no_holdout_in_prompts([rendered], rid)
            rendered_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            rendered_hashes = [rendered_hash]
            if not live:
                output = ""
                status = "not_live"
                usage = {}
                http_status = 200
                latency_ms = 0
            else:
                t0 = time.time()
                try:
                    resp = chat(messages, **params)
                    output = resp.text
                    usage = resp.usage or {}
                    http_status = resp.status
                    status = "answered" if resp.status == 200 else "error"
                except LLMError as exc:
                    output = ""
                    usage = {}
                    http_status = None
                    status = "error"
                    _ = str(exc)
                latency_ms = int((time.time() - t0) * 1000)
            context_pages = sorted([list(k) for k in context_keys])
            record_extra = {}

        if "record_extra" not in locals():
            record_extra = {}

        record = {
            "run_id": run_id, "arm": arm, "row_id": rid,
            "flow_bucket": row.get("flow_bucket"),
            "status": status, "http_status": http_status,
            "output": output, "usage": usage, "latency_ms": latency_ms,
            "context_pages": context_pages,
            "rendered_prompt_hashes": rendered_hashes,
            "rendered_prompt_hash": (rendered_hashes[0] if rendered_hashes else None),
            "prompt_set_hash": prompt_set_hash(),
            "model_params_hash": params_hash,
            "label": "SMOKE_NOT_SCORED" if smoke else "SCORED",
        }
        record.update(record_extra)
        record["score"] = score_prediction(row, output, set(tuple(x) for x in context_pages))

        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

        processed += 1
        if status == "error":
            errors += 1
        if processed and errors / processed > 0.05:
            stopped_reason = f"error rate {errors}/{processed} exceeded 5%"
            break

    manifest.update({
        "finished_at": datetime.now(UTC).isoformat(),
        "n_processed_this_run": processed,
        "n_errors": errors,
        "error_rate": (errors / processed) if processed else 0.0,
        "stopped_reason": stopped_reason,
        "output_path": str(out_path),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
