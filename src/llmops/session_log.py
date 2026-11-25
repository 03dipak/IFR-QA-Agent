from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from src.safety import mask_value


def write_session_log(
    question: str,
    answer: str,
    citations: list[str],
    latency_s: float,
    steps: list[dict[str, Any]],
    *,
    session_id: str,
    error: str = "",
    stop_reason: str = "",
    error_type: str = "none",
    root: Path | str | None = None,
    status: str = "ok",
) -> Path:
    base = Path(root) if root is not None else Path(
        os.environ.get("AGENT_SESSION_LOG_DIR", "data/sessions")
    )
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in session_id)
    safe = safe or "default"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{safe}.jsonl"
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_id": session_id,
        "question": mask_value(question),
        "answer": mask_value(answer),
        "citations": mask_value(citations),
        "latency_s": round(float(latency_s), 3),
        "steps": mask_value(steps),
        "error": mask_value(error),
        "stop_reason": stop_reason,
        "error_type": error_type,
        "status": status,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return path
