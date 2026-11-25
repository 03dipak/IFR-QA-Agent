"""Executor agent: run plan steps with caps, loop detection and exception observations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.llmops.tracing import Tracer
from src.tools import calculator, doc_lookup, retrieve


@dataclass
class Observation:
    step: int
    sub_question: str
    tool: str
    args: dict[str, Any]
    text: str
    latency_ms: int
    error: bool = False
    chunks: list[Any] = field(default_factory=list)


class Executor:
    def __init__(self, tools: dict[str, Callable[..., Any]] | None = None,
                 *, max_steps: int = 10, max_tokens: int = 4000,
                 tracer: Tracer | None = None) -> None:
        self.tools = tools or {
            "retrieve": retrieve.retrieve,
            "calculator": calculator.calculate,
            "doc_lookup": doc_lookup.lookup,
        }
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.tracer = tracer

    def run(self, steps: list[Any], *, tokens_used: int = 0) -> list[Observation]:
        observations: list[Observation] = []
        seen: set[tuple[str, str]] = set()
        for position, step in enumerate(steps[:self.max_steps], start=1):
            if tokens_used >= self.max_tokens:
                break
            tool = step.tool
            args = dict(step.args or {})
            key = (tool, json_key(args))
            if key in seen:
                observations.append(Observation(
                    step=position, sub_question=step.sub_question, tool=tool, args=args,
                    text=f"STOP: repeated tool+args ({tool} {args}); loop detected.",
                    latency_ms=0, error=True,
                ))
                if self.tracer:
                    self.tracer.record("tool", tool=tool, detail="loop_detected")
                break
            seen.add(key)
            start = time.monotonic()
            try:
                result = self.tools[tool](**args)
                text = str(result)
                error = False
                chunks = result if tool == "retrieve" and isinstance(result, list) else []
            except Exception as exc:
                result = None
                text = f"TOOL ERROR: {type(exc).__name__}: {exc}"
                error = True
                chunks = []
            latency = int((time.monotonic() - start) * 1000)
            observations.append(Observation(
                step=position, sub_question=step.sub_question, tool=tool, args=args,
                text=text, latency_ms=latency, error=error, chunks=chunks,
            ))
            if self.tracer:
                self.tracer.record("tool", tool=tool, latency_ms=latency, error=error)
        if len(steps) > self.max_steps and self.tracer:
            self.tracer.record("cap", detail=f"step_cap:{self.max_steps}")
        return observations


def json_key(args: dict[str, Any]) -> str:
    import json
    return json.dumps(args, sort_keys=True, default=str)
