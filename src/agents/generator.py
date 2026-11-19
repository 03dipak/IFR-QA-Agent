"""Generator agent: produces a draft answer for a query."""

from __future__ import annotations

from src.agents import prompts
from src.agents.llm_client import Completion, LLMClient


class Generator:
    """Actor that turns a query into a first-draft answer."""

    def __init__(self, client: LLMClient, max_tokens: int = 512) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def generate(self, query: str) -> tuple[str, Completion]:
        """Return (draft_text, completion usage) for the query."""
        completion = self._client.chat(
            prompts.GENERATOR_SYSTEM, prompts.generator_user(query), self._max_tokens,
        )
        return completion.text, completion
