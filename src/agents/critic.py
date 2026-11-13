"""Critic agent: returns a validated strict-JSON critique of a draft."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from src.agents import prompts
from src.agents.llm_client import Completion, LLMClient

logger = logging.getLogger(__name__)


class CriticParseError(ValueError):
    """Critic output was not parseable strict JSON."""


class Critique(BaseModel):
    """Validated critic output."""
    score: int = Field(ge=1, le=5)
    issues: list[str]
    verdict: Literal["pass", "revise"]


@dataclass(frozen=True)
class CritiqueOutcome:
    """Critique plus per-step usage metadata."""
    critique: Critique
    completion: Completion
    parse_retries: int


def parse_critique(text: str) -> Critique:
    """Parse strict JSON critique, tolerating code fences."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        body = body.removeprefix("json").strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise CriticParseError(f"critic output is not JSON: {exc}") from exc
    try:
        return Critique.model_validate(data)
    except ValidationError as exc:
        raise CriticParseError(f"critic JSON failed schema: {exc}") from exc


class Critic:
    """Evaluator scoring the draft; falls back to verdict 'pass' after one retry."""

    def __init__(self, client: LLMClient, max_tokens: int = 256) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def review(self, query: str, draft: str) -> CritiqueOutcome:
        user = prompts.critic_user(query, draft)
        completion = self._client.chat(prompts.CRITIC_SYSTEM, user, self._max_tokens)
        try:
            critique = parse_critique(completion.text)
            return CritiqueOutcome(critique, completion, parse_retries=0)
        except CriticParseError as first_error:
            logger.warning("critic JSON malformed, retrying once: %s", first_error)
            retry = self._client.chat(
                prompts.CRITIC_SYSTEM, user + "\nReturn ONLY the JSON object.", self._max_tokens,
            )
            try:
                critique = parse_critique(retry.text)
                return CritiqueOutcome(critique, retry, parse_retries=1)
            except CriticParseError as second_error:
                logger.error("critic JSON malformed twice, falling back to pass: %s", second_error)
                fallback = Critique(score=1, issues=["critic output unparseable"], verdict="pass")
                return CritiqueOutcome(fallback, retry, parse_retries=1)
