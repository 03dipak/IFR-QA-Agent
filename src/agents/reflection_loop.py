"""Reflection loop orchestrator: generate -> critic -> (revise -> critic) with budget cap."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.agents.critic import Critic
from src.agents.generator import Generator
from src.agents.llm_client import Completion, LLMClient
from src.agents.reviser import Reviser
from src.config import load_reflection_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceStep:
    """One stage of the loop with its usage."""
    iteration: int
    stage: str
    text: str
    critique: dict | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


@dataclass(frozen=True)
class ReflectionResult:
    """Final answer plus the full trace."""
    final_answer: str
    iterations: int
    passed: bool
    stopped_reason: str
    total_tokens: int
    trace: list[TraceStep] = field(default_factory=list)


def _add_step(
    trace: list[TraceStep],
    iteration: int,
    stage: str,
    text: str,
    completion: Completion,
    critique: dict | None = None,
) -> None:
    trace.append(TraceStep(
        iteration=iteration, stage=stage, text=text, critique=critique,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        latency_ms=completion.latency_ms,
    ))


def _tokens_used(trace: list[TraceStep]) -> int:
    return sum(step.prompt_tokens + step.completion_tokens for step in trace)


def run_reflection(
    client: LLMClient,
    query: str,
    *,
    max_iterations: int | None = None,
    max_total_tokens: int | None = None,
) -> ReflectionResult:
    """Run the loop; stop on critic pass, max iterations, or token-budget cap."""
    config = load_reflection_config()
    max_iterations = max_iterations if max_iterations is not None else config.max_iterations
    max_total_tokens = (
        max_total_tokens if max_total_tokens is not None else config.max_total_tokens
    )

    generator = Generator(client)
    critic = Critic(client)
    reviser = Reviser(client)
    trace: list[TraceStep] = []
    draft: str | None = None

    for iteration in range(1, max_iterations + 1):
        if _tokens_used(trace) >= max_total_tokens:
            break
        if draft is None:
            draft, completion = generator.generate(query)
            _add_step(trace, iteration, "generate", draft, completion)
        outcome = critic.review(query, draft)
        _add_step(
            trace, iteration, "critique", outcome.completion.text, outcome.completion,
            critique=outcome.critique.model_dump(),
        )
        if outcome.critique.verdict == "pass":
            return _result(draft, iteration, True, "critic_pass", trace)
        if iteration == max_iterations:
            break
        revised, completion = reviser.revise(query, draft, outcome.critique.issues)
        _add_step(trace, iteration, "revise", revised, completion)
        draft = revised

    reason = "token_budget" if _tokens_used(trace) >= max_total_tokens else "max_iterations"
    logger.info(
        "reflection stopped: %s iterations=%d tokens=%d",
        reason, len(trace), _tokens_used(trace),
    )
    return _result(draft or "", max_iterations, False, reason, trace)


def _result(
    final_answer: str, iterations: int, passed: bool, reason: str, trace: list[TraceStep],
) -> ReflectionResult:
    return ReflectionResult(
        final_answer=final_answer, iterations=iterations, passed=passed,
        stopped_reason=reason, total_tokens=_tokens_used(trace), trace=trace,
    )
