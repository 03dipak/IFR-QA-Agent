"""System baselines B0/B1/B2 with pinned model parameters.

Pinned params: thinking off, 8,192-token ceiling, output cap 512, retry
2/4/8s, concurrency 1 (enforced by src.llm.client). Smoke outputs are
labelled SMOKE_NOT_SCORED and are never used as scores.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.llm.client import chat
from src.eval.prompts import build_messages, prompt_set_hash, render_prompt

MODEL_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 512,
    "disable_thinking": True,
    "cache": False,
    "timeout": 60.0,
}

PINNED_CEILING = 8192


def _call(messages):
    from src.llm.client import MAX_TOTAL_TOKENS
    if MAX_TOTAL_TOKENS != PINNED_CEILING:
        raise RuntimeError(f"endpoint ceiling changed: {MAX_TOTAL_TOKENS}")
    return chat(messages, **MODEL_PARAMS)


def run_baseline(baseline: str, row: dict, *, context: str | None = None,
                 few_shot_rows: list[dict] | None = None,
                 retrieval_variant: str = "global", live: bool = False) -> dict:
    messages = build_messages(baseline, row, context=context,
                              few_shot_rows=few_shot_rows,
                              retrieval_variant=retrieval_variant)
    out = {
        "label": "SMOKE_NOT_SCORED",
        "baseline": baseline,
        "retrieval_variant": retrieval_variant,
        "row_id": row.get("row_id") or row.get("id"),
        "prompt_set_version": __import__("src.eval.prompts", fromlist=["PROMPT_SET_VERSION"]).PROMPT_SET_VERSION,
        "prompt_set_hash": prompt_set_hash(),
        "rendered_prompt": render_prompt(messages),
        "live": bool(live),
        "model_params": dict(MODEL_PARAMS),
        "token_ceiling": PINNED_CEILING,
    }
    if not live:
        out["output"] = ""
        out["status"] = "not_live"
        return out
    t0 = time.time()
    resp = _call(messages)
    latency_ms = int((time.time() - t0) * 1000)
    out.update({
        "output": resp.text,
        "status": "answered" if resp.status == 200 else "error",
        "http_status": resp.status,
        "prompt_tokens": resp.usage.get("prompt_tokens"),
        "completion_tokens": resp.usage.get("completion_tokens"),
        "latency_ms": latency_ms,
        "prompt_version": resp.prompt_version,
        "model": __import__("os").environ.get("LLM_MODEL", "Qwen/Qwen3-8B"),
    })
    return out


def run_b0(row: dict, *, few_shot_rows: list[dict] | None = None,
           live: bool = False) -> dict:
    return run_baseline("B0", row, few_shot_rows=few_shot_rows, live=live)


def run_b1(row: dict, *, context: str | None = None, live: bool = False) -> dict:
    return run_baseline("B1", row, context=context, live=live)


def run_b2(row: dict, *, context: str | None = None,
           retrieval_variant: str = "global", live: bool = False) -> dict:
    return run_baseline("B2", row, context=context,
                        retrieval_variant=retrieval_variant, live=live)
