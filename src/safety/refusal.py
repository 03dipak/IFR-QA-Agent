"""Week 1 refusal / human-escalation policy for Track B (no new gate elsewhere)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.citations.models import VerificationOutcome
from src.safety import detect_injection, validate_tool_args

_ADVICE = re.compile(r"\b(buy|sell|hold|should i|recommend(?:ation)?|invest in)\b", re.I)
_FORECAST = re.compile(r"\b(forecast|predict|will be|projection|future price|target price)\b", re.I)
_LIVE = re.compile(r"\b(live|current price|market price|today's price|share price now)\b", re.I)


@dataclass(frozen=True)
class Refusal:
    refused: bool
    reason: str = ""
    escalate: bool = False


def check_intent(question: str) -> Refusal:
    """Pre-retrieval refusals from docs/WEEK1_COPILOT_SPEC.md section 2."""
    if _ADVICE.search(question or ""):
        return Refusal(
            True, "Investment advice is out of scope (no buy/sell/hold recommendations).",
        )
    if _FORECAST.search(question or ""):
        return Refusal(True, "The system does not produce its own forecasts or projections.")
    if _LIVE.search(question or ""):
        return Refusal(True, "Live or current market prices are not available.")
    if detect_injection(question or ""):
        return Refusal(True, "Prompt injection signal detected in the question.", escalate=True)
    return Refusal(False)


def check_evidence(has_evidence: bool, reason: str = "") -> Refusal:
    """Post-retrieval abstention / escalation."""
    if has_evidence:
        return Refusal(False)
    return Refusal(
        True,
        reason or "Insufficient in-corpus evidence; abstaining rather than fabricating a figure.",
        escalate=False,
    )


def from_verification(outcome: VerificationOutcome) -> Refusal:
    if outcome is VerificationOutcome.REQUIRES_HUMAN_REVIEW:
        return Refusal(True, "Citation requires human review.", escalate=True)
    if outcome in (VerificationOutcome.UNVERIFIABLE, VerificationOutcome.INSUFFICIENT_EVIDENCE):
        return Refusal(True, f"Citation could not be verified: {outcome.value}.", escalate=True)
    return Refusal(False)


__all__ = ["Refusal", "check_intent", "check_evidence", "from_verification", "validate_tool_args"]
