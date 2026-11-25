"""Planner agent: JSON-validated tool plan with two malformed-JSON retries."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.agents.llm_client import Completion, LLMClient

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = (
    "You are the Planner for a financial document QA agent. Reply with ONLY a JSON object:\n"
    '{"steps": [{"sub_question": "<one focused question per company/document>", '
    '"tool": "retrieve" | "doc_lookup" | "calculator", "args": {"query": "<text>"}}]}\n'
    "Use one retrieve step per company/document; args.query must be the sub-question text. "
    "Use doc_lookup only to list matching documents, calculator only for arithmetic. "
    "Every factual question needs at least one retrieve step. No document ids in the output."
)


class PlanStep(BaseModel):
    sub_question: str = Field(min_length=1)
    tool: Literal["retrieve", "doc_lookup", "calculator"]
    args: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1)


def parse_plan(text: str) -> Plan:
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    return Plan.model_validate(json.loads(body))


class Planner:
    def __init__(self, client: LLMClient, max_tokens: int = 512, retries: int = 2) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._retries = retries

    def plan(self, question: str) -> tuple[Plan, list[Completion], int]:
        user = f"Question: {question}\n\nJSON plan:"
        completions: list[Completion] = []
        error = ""
        for attempt in range(self._retries + 1):
            prompt = user + (f"\nPrevious output invalid: {error}\nReturn ONLY valid JSON."
                             if error else "")
            completion = self._client.chat(PLANNER_SYSTEM, prompt, self._max_tokens)
            completions.append(completion)
            try:
                return parse_plan(completion.text), completions, attempt
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                error = str(exc)[:400]
                logger.warning("plan JSON invalid (attempt %d): %s", attempt + 1, error)
        fallback = Plan(steps=[PlanStep(
            sub_question=question, tool="retrieve", args={"query": question},
        )])
        return fallback, completions, self._retries
