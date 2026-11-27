"""Intent-based refusals for the QA agent."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.citations.models import VerificationOutcome
from src.safety import detect_injection, validate_tool_args

INVESTMENT_ADVICE_REPLY = (
    "I can't recommend or advise on buying, selling or choosing stocks, mutual funds or other investments. "
    "I can report and compare figures from the company annual reports available here, such as revenue, profit margins or total assets."
)
OUT_OF_SCOPE_REPLY = (
    "This question is outside my scope. I answer questions using the Indian company annual reports available here. "
    "Try asking about a company, financial year and figure."
)

# advisory intent, not mere mention of buy/fund/stock in factual report questions
_ADVICE_PATTERNS = [
    re.compile(r"\bshould\s+i\s+(buy|sell|hold|invest|purchase)\b", re.I),
    re.compile(r"\b(is|are)\s+.+\s+(good|bad|worth)\s+to\s+(buy|sell|hold|invest)\b", re.I),
    re.compile(r"\b(recommend|suggest|advise)\b.*\b(stock|stocks|share|shares|mutual fund|mutual funds|etf|etfs|investment|investments|portfolio)\b", re.I),
    re.compile(r"\b(which|what)\b.*\b(stock|stocks|share|shares|mutual fund|mutual funds|etf|etfs)\b.*\b(buy|invest|pick|choose|good)\b", re.I),
    re.compile(r"\b(buy|sell|hold|invest in|purchase)\b.*\b(stock|stocks|share|shares|mutual fund|mutual funds|etf|etfs)\b", re.I),
    re.compile(r"\b(good|best|top)\b.*\b(stock|stocks|share|shares|mutual fund|mutual funds)\b.*\b(to buy|to invest|for investment|right now|now)\b", re.I),
]
_FORECAST = re.compile(r"\b(forecast|predict|projection|target price|future price|will .* price)\b", re.I)
_LIVE = re.compile(r"\b(live|current price|market price|today'?s price|share price now|stock price now)\b", re.I)

# factual report terms that should not trigger investment-advice refusal by themselves
_FACTUAL_REPORT_TERMS = re.compile(
    r"\b(buyback|buy\s+back|share buyback|shares?\s+repurchas\w*|repurchase of shares|mutual fund investments?|investments? in mutual funds?|"
    r"inventor(?:y|ies)|stock[- ]in[- ]trade|stock of raw materials?|funds? flow|funded debt|"
    r"treasury stock|stock compensation|stock option|employee stock|stock split|bonus share)\b",
    re.I,
)

_FINANCIAL_REPORT_HINTS = re.compile(
    r"\b(annual report|consolidated|standalone|fy\d{4}|fiscal year|year ended|march 31|march 202\d|"
    r"revenue|profit|loss|assets?|liabilities|equity|cash flow|balance sheet|income statement|"
    r"dividend declared|dividend paid|eps|ebitda|margin|inventor(?:y|ies)|receivables?|borrowings?|"
    r"buyback amount|share buyback|mutual fund investments?|notes to accounts|auditor|segment|capex)\b",
    re.I,
)
_COMPANY_LIKE = re.compile(r"\b[A-Z][A-Za-z&.\-]{2,}\b")


@dataclass(frozen=True)
class Refusal:
    refused: bool
    reason: str = ""
    escalate: bool = False
    category: str = ""


def _is_investment_advice(question: str) -> bool:
    q = question or ""
    if not q.strip():
        return False
    # If it's clearly asking for report facts, don't refuse just because it mentions buy/fund/stock.
    has_factual_report_term = bool(_FACTUAL_REPORT_TERMS.search(q))
    has_advice_pattern = any(p.search(q) for p in _ADVICE_PATTERNS)
    if not has_advice_pattern:
        return False
    if has_factual_report_term:
        # refuse only if explicit advisory phrasing is present
        explicit = re.compile(r"\b(should i|recommend|suggest|advise|good to buy|worth buying|which .* to buy|which .* to invest)\b", re.I)
        return bool(explicit.search(q))
    return True


def _is_out_of_scope(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if _FINANCIAL_REPORT_HINTS.search(q):
        return False
    # allow short conversational inputs handled elsewhere
    if len(q.split()) <= 4:
        return False
    # General knowledge / unrelated questions: no company-report intent signals.
    # If it has company-like proper noun but no financial/report hint, treat as out of scope for this agent.
    return True


def check_intent(question: str) -> Refusal:
    q = question or ""
    if _is_investment_advice(q):
        return Refusal(True, INVESTMENT_ADVICE_REPLY, category="investment_advice")
    if _FORECAST.search(q) and not _FINANCIAL_REPORT_HINTS.search(q):
        return Refusal(True, OUT_OF_SCOPE_REPLY, category="out_of_scope")
    if _LIVE.search(q) and not _FINANCIAL_REPORT_HINTS.search(q):
        return Refusal(True, OUT_OF_SCOPE_REPLY, category="out_of_scope")
    if detect_injection(q):
        return Refusal(True, "I can’t help with that request.", escalate=True, category="unsafe")
    if _is_out_of_scope(q):
        return Refusal(True, OUT_OF_SCOPE_REPLY, category="out_of_scope")
    return Refusal(False)


def check_evidence(has_evidence: bool, reason: str = "") -> Refusal:
    if has_evidence:
        return Refusal(False)
    return Refusal(True, reason or "I could not find that figure in the available annual reports.", escalate=False, category="insufficient_evidence")


def from_verification(outcome: VerificationOutcome) -> Refusal:
    if outcome is VerificationOutcome.REQUIRES_HUMAN_REVIEW:
        return Refusal(True, "Citation requires human review.", escalate=True, category="verification")
    if outcome in (VerificationOutcome.UNVERIFIABLE, VerificationOutcome.INSUFFICIENT_EVIDENCE):
        return Refusal(True, f"Citation could not be verified: {outcome.value}.", escalate=True, category="verification")
    return Refusal(False)


__all__ = [
    "Refusal", "check_intent", "check_evidence", "from_verification", "validate_tool_args",
    "INVESTMENT_ADVICE_REPLY", "OUT_OF_SCOPE_REPLY",
]
