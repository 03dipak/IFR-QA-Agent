"""Reviser agent: improves a draft using the critic's issues."""

from __future__ import annotations

from src.agents import prompts
from src.agents.llm_client import Completion, LLMClient


class Reviser:
    """Refiner that rewrites the draft to address critic issues."""

    def __init__(self, client: LLMClient, max_tokens: int = 512) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def revise(self, query: str, draft: str, issues: list[str]) -> tuple[str, Completion]:
        """Return (revised_text, completion usage)."""
        completion = self._client.chat(
            prompts.REVISER_SYSTEM, prompts.reviser_user(query, draft, issues), self._max_tokens,
        )
        return completion.text, completion
