"""Bounded per-request deadline and retry limits (lld_agents.txt, lld_api_ui.txt).

Every external call asks this helper for its timeout, so no stage can outlive
the request budget even when an endpoint stalls.
"""

from __future__ import annotations

import os
import threading
import time

_local = threading.local()


def start(budget_s: float) -> float:
    """Start a deadline for the current worker thread."""
    deadline = time.monotonic() + float(budget_s)
    _local.deadline = deadline
    return deadline


def finish() -> None:
    """Clear the deadline at the end of a request."""
    _local.deadline = None


def check() -> None:
    """Raise TimeoutError if the request budget is gone."""
    deadline: float | None = getattr(_local, "deadline", None)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("request deadline exceeded")


def soft_check(margin_s: float = 20.0) -> None:
    """Raise TimeoutError when only the soft-deadline margin remains."""
    remaining_s = getattr(_local, "deadline", None)
    if remaining_s is None:
        return
    left = float(remaining_s) - time.monotonic()
    if left <= 0:
        raise TimeoutError("request deadline exceeded")
    if left <= float(margin_s):
        raise TimeoutError(f"soft deadline reached ({left:.1f}s left)")


def remaining(default_s: float) -> float:
    """Return the smaller of the call timeout and the remaining budget."""
    check()
    deadline: float | None = getattr(_local, "deadline", None)
    if deadline is None:
        return float(default_s)
    return min(float(default_s), float(deadline) - time.monotonic())


def sleep(seconds: float) -> None:
    """Backoff sleep that never crosses the request deadline."""
    check()
    deadline: float | None = getattr(_local, "deadline", None)
    if deadline is not None and time.monotonic() + seconds >= float(deadline):
        raise TimeoutError("request deadline exceeded")
    time.sleep(seconds)


def _env_float(name: str) -> float:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return float(value)


def llm_timeout_s() -> float:
    return _env_float("LLM_CALL_TIMEOUT_SECONDS")


def embed_timeout_s() -> float:
    return _env_float("EMBED_CALL_TIMEOUT_SECONDS")


def max_retries() -> int:
    value = os.environ.get("EXTERNAL_CALL_MAX_RETRIES")
    if not value:
        raise RuntimeError("EXTERNAL_CALL_MAX_RETRIES is not configured")
    retries = int(value)
    if not 0 <= retries <= 2:
        raise ValueError("EXTERNAL_CALL_MAX_RETRIES must be between 0 and 2")
    return retries


def backoff_s(attempt: int) -> float:
    value = os.environ.get("EXTERNAL_CALL_BACKOFF_SECONDS")
    if not value:
        raise RuntimeError("EXTERNAL_CALL_BACKOFF_SECONDS is not configured")
    delays = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not delays:
        raise ValueError("EXTERNAL_CALL_BACKOFF_SECONDS must contain a delay")
    return delays[min(attempt, len(delays) - 1)]
