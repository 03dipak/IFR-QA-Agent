"""Step tracing for the Track B agent layer (no prior tracing existed in src/llmops)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

TRACE_ROOT = Path("logs/traces")


def trace_path(session_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "default"
    return TRACE_ROOT / f"{safe}.jsonl"


class Tracer:
    """Append-only JSONL trace: tool, args, latency and tokens per step."""

    def __init__(self, session_id: str, root: Path | str = TRACE_ROOT) -> None:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "default"
        self.path = Path(root) / f"{safe}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        stage: str,
        *,
        tool: str = "",
        args: dict[str, Any] | None = None,
        latency_ms: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        detail: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id": self.path.stem,
            "stage": stage,
            "tool": tool,
            "args": args or {},
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens": prompt_tokens + completion_tokens,
            "detail": detail,
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return row
