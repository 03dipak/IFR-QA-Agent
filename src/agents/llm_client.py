"""Thin chat interface for reflection agents, backed by the single endpoint client."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from src.llm import client as endpoint


@dataclass(frozen=True)
class Completion:
    """One chat completion result with token usage."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class LLMClient(Protocol):
    """Minimal interface the reflection agents depend on."""

    def chat(self, system: str, user: str, max_tokens: int) -> Completion: ...


class EndpointLLMClient:
    """OpenAI-compatible endpoint adapter (env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model

    def chat(self, system: str, user: str, max_tokens: int) -> Completion:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        start = time.monotonic()
        # cache=False: reflection steps must see fresh generations, not cached drafts.
        response = endpoint.chat(
            messages, model=self._model, max_tokens=max_tokens, cache=False,
        )
        usage = response.usage or {}
        return Completion(
            text=endpoint.strip_thinking(response.text),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=int((time.monotonic() - start) * 1000),
        )
