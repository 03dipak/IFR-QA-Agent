from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import requests


class LLM(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 512) -> str: ...


class StubLLM:
    def __init__(self, responses: Iterable[str] | Callable[[str], str]):
        self._responses = None if callable(responses) else deque(responses)
        self._callable = responses if callable(responses) else None
        self.calls: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        self.calls.append(prompt)
        if self._callable is not None:
            return self._callable(prompt)
        if not self._responses:
            if "keys answer, citations, insufficient_evidence" in prompt:
                return '{"answer":"ok","citations":[],"insufficient_evidence":false}'
            raise RuntimeError("StubLLM has no responses left")
        return self._responses.popleft()


class EnvLLM:
    def __init__(self) -> None:
        self.url = os.environ.get("LLM_CHAT_URL")
        self.model = os.environ.get("LLM_MODEL")
        self.api_key = os.environ.get("LLM_API_KEY")
        self.ca_cert = os.environ.get("LLM_CA_CERT")
        if not self.url or not self.model:
            raise RuntimeError("LLM_CHAT_URL and LLM_MODEL must be configured")

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        verify: str | bool = True
        if self.ca_cert:
            ca_path = Path(self.ca_cert)
            if ca_path.is_file():
                verify = str(ca_path)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            timeout=120,
            verify=verify,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class TokenBudget:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.used = 0

    def reset(self) -> None:
        self.used = 0

    def add(self, text: str) -> None:
        self.used += max(1, len(text) // 4)

    @property
    def exceeded(self) -> bool:
        return self.used >= self.max_tokens
