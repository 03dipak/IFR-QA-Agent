"""Pinned model parameters and their hashes for Week-1 prompt runs."""

from __future__ import annotations

import hashlib
import json

MODEL = "Qwen/Qwen3-8B"
TEMPERATURE = 0.0
SELF_CONSISTENCY_TEMPERATURE = 0.7
TOP_P = 1.0
MAX_TOKENS = 512
COT_MAX_TOKENS = 1024
CEILING = 8192
CONCURRENCY = 1
RATE_LIMIT_PER_MIN = 60
RETRY_BACKOFF_S = (2, 4, 8)


def model_params(temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS) -> dict:
    return {
        "model": MODEL,
        "temperature": temperature,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        "disable_thinking": True,
        "cache": False,
        "timeout": 60.0,
    }


def model_params_hash(temperature: float = TEMPERATURE,
                      max_tokens: int = MAX_TOKENS) -> str:
    payload = model_params(temperature=temperature, max_tokens=max_tokens)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
