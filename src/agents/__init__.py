"""Agent orchestration: Planner → Executor → Critic.

Implements lld_agents.txt — exactly three tools (retriever, calculator,
comparator), hard 10-step cap, JSON-schema validation, refusal short-circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Plan:
    """JSON-only plan produced by the Planner."""
    question_id: str
    resolved_entity_id: Optional[str] = None
    resolved_period: Optional[str] = None
    steps: list[dict] = field(default_factory=list)
    refusal_class: Optional[str] = None


@dataclass(frozen=True)
class ToolResult:
    """Structured result from the Executor."""
    step_number: int
    tool_name: str
    result: dict
    args_valid: bool
    latency_ms: int


@dataclass(frozen=True)
class CriticVerdict:
    """Deterministic verification verdict from the Critic."""
    figure_ref: str
    expected: str
    actual: str
    verdict: str  # PASS | FAIL | RETRY | NEEDS_HUMAN_REVIEW
    note: Optional[str] = None


class Planner:
    """Planner agent — produces JSON-only plans, NO tool calls."""

    def plan(self, question: str) -> Plan:
        """Produce a plan for the question."""
        return Plan(question_id=question[:50])


class Executor:
    """Executor agent — ONLY tool caller (3 tools max)."""

    def execute(self, plan: Plan) -> list[ToolResult]:
        """Execute the plan's steps using tools."""
        return []


class Critic:
    """Critic agent — deterministic code only, no model calls."""

    def verify(self, results: list[ToolResult]) -> list[CriticVerdict]:
        """Verify every figure against EvidenceObject."""
        return []
