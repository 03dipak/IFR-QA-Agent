from __future__ import annotations

from typing import Any, TypedDict

StopReason = str


class AgentState(TypedDict, total=False):
    query: str
    requested: list[dict[str, str]]
    retrieval_round: int
    fallback_used: bool
    plan: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    draft_answer: str
    critique: dict[str, Any]
    iteration: int
    steps_used: int
    errors: list[str]
    stop_reason: StopReason
    token_count: int
    final_answer: dict[str, Any]
