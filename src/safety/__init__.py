"""Input and tool safety guards for the financial QA agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

APPROVAL_REQUIRED = "Requires human approval"
ALLOWED_TOOLS = frozenset({"search_index", "get_chunk", "calculator", "list_companies", "get_metric", "rank_values", "rank_companies", "sum_values", "compare_periods"})
_PII_MASK = "[REDACTED]"


@dataclass(frozen=True)
class InjectionSignal:
    """Detected injection signal."""

    signal_type: str
    description: str
    severity: str


@dataclass(frozen=True)
class SafetyDecision:
    """Result of checking one user question."""

    refused: bool
    reason: str = ""
    escalate: bool = False
    sanitized: str = ""
    injection: tuple[InjectionSignal, ...] = ()
    pii: bool = False


_INJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above|your)\s+"
                r"(?:instructions|rules|citation rules)", re.I),
     "ignore_instructions", "Ignore-previous-instructions pattern"),
    (re.compile(r"\b(?:SYSTEM|ROLE|ASSISTANT)\s*:", re.I),
     "role_marker", "Role-marker escalation pattern"),
    (re.compile(r"\b(system prompt|developer message|hidden instructions?)\b", re.I),
     "prompt_exfiltration", "System/developer prompt request"),
    (re.compile(r"\b(env(?:ironment)? (?:keys?|vars?)|api[_ -]?key|secret[_ -]?key|\.env)\b", re.I),
     "secret_exfiltration", "Environment or key request"),
    (re.compile(
        r"\b(?:call|invoke|use|run|execute)\s+(?:the\s+)?(?:tool\s+)?"
        r"(?:send_email|delete_all|shell|exec|browser|unknown_tool)\b", re.I,
    ),
     "unknown_tool", "Unknown tool request"),
    (re.compile(r"\b(?:use|call|invoke|run)\b[^\n]{0,40}\b(?:send|email|delete|transfer)\b", re.I),
     "tool_misuse", "Tool misuse pattern"),
    (re.compile(
        r"\b(retrieved chunk|document|pdf|note)\s+(?:says|contains|instructs)\b"
        r".*\b(ignore|disregard|forget|reveal|execute)\b", re.I,
    ),
     "injected_chunk", "Instruction-like text in retrieved data"),
)
_CONSEQUENTIAL = re.compile(
    r"\b(approve|authorize|make|send|transfer|wire|pay|delete|submit|sign)\b"
    r".*\b(payment|email|money|funds|invoice|report|account)\b",
    re.I,
)
_OUT_OF_SCOPE = re.compile(
    r"\b(should i|buy|sell|hold|invest in|forecast|predict|target price|stock price)\b",
    re.I,
)
_PERSONAL_REQUEST = re.compile(
    r"\b(?:give|show|provide|tell|what is|what's|where is)\s+(?:me\s+)?(?:the\s+)?"
    r"(personal (?:data|details|information)|phone number|mobile number|"
    r"aadhaar|pan|passport|home address|date of birth|dob)\b",
    re.I,
)
_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ACCOUNT = re.compile(r"(?i)\b(?:account|a/c|ifsc)\b[^\n]{0,40}\b\d{9,18}\b")


def mask_pii(text: str) -> str:
    """Replace common Indian PII identifiers with a fixed token."""
    masked = _PAN.sub(_PII_MASK, str(text))
    masked = _AADHAAR.sub(_PII_MASK, masked)
    masked = _PHONE.sub(_PII_MASK, masked)
    masked = _EMAIL.sub(_PII_MASK, masked)
    def _mask_account(match: re.Match[str]) -> str:
        text = match.group(0)
        digits = re.search(r"\d{9,18}", text)
        return text.replace(digits.group(0), _PII_MASK, 1) if digits else text
    return _ACCOUNT.sub(_mask_account, masked)


def mask_value(value: Any) -> Any:
    """Recursively mask PII in logs and trace payloads."""
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {key: mask_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_value(item) for item in value]
    return value


def detect_injection(text: str) -> list[InjectionSignal]:
    """Detect prompt-injection signals in user text or retrieved data."""
    found: list[InjectionSignal] = []
    for pattern, signal_type, description in _INJECTION_PATTERNS:
        if pattern.search(text or ""):
            found.append(InjectionSignal(signal_type, description, "high"))
    return found


def check_input(text: str) -> SafetyDecision:
    """Refuse unsafe input and return a PII-masked copy for logs/traces."""
    original = text or ""
    sanitized = mask_pii(original)
    pii = sanitized != original
    signals = detect_injection(original)
    if signals:
        return SafetyDecision(
            True, "Prompt injection signal detected; request refused.", True,
            sanitized, tuple(signals), pii,
        )
    if _CONSEQUENTIAL.search(original):
        return SafetyDecision(True, APPROVAL_REQUIRED, True, sanitized, (), pii)
    if _OUT_OF_SCOPE.search(original):
        return SafetyDecision(
            True, "Out of scope for this financial-report QA system.", False, sanitized, (), pii,
        )
    if _PERSONAL_REQUEST.search(original):
        return SafetyDecision(
            True, "Personal data requests are out of scope.", True, sanitized, (), pii,
        )
    return SafetyDecision(False, "", False, sanitized, (), pii)


def is_pdf_data_not_instruction(text: str) -> bool:
    """Treat PDF content as DATA, never instructions."""
    return not detect_injection(text)


def validate_tool_args(tool_name: str, args: dict) -> bool:
    """Validate a tool name and its arguments against the allowed list."""
    if tool_name not in ALLOWED_TOOLS or not isinstance(args, dict):
        return False
    if tool_name == "search_index":
        query = args.get("query")
        return isinstance(query, str) and bool(query.strip())
    if tool_name == "get_chunk":
        chunk_id = args.get("chunk_id")
        return isinstance(chunk_id, str) and bool(chunk_id.strip())
    if tool_name == "calculator":
        expression = args.get("expression")
        return isinstance(expression, str) and 0 < len(expression.strip()) <= 200
    if tool_name == "list_companies":
        return True
    if tool_name == "get_metric":
        return isinstance(args.get("company"), str) and bool(str(args.get("company")).strip()) and isinstance(args.get("metric"), str) and bool(str(args.get("metric")).strip())
    if tool_name == "rank_values":
        return isinstance(args.get("items"), list)
    if tool_name == "rank_companies":
        return isinstance(args.get("metric"), str) and bool(str(args.get("metric")).strip())
    if tool_name == "sum_values":
        return isinstance(args.get("items"), list)
    if tool_name == "compare_periods":
        return isinstance(args.get("metric"), str) and bool(str(args.get("metric")).strip()) and isinstance(args.get("base_period"), str) and isinstance(args.get("compare_period"), str)
    return False


__all__ = [
    "ALLOWED_TOOLS",
    "APPROVAL_REQUIRED",
    "InjectionSignal",
    "SafetyDecision",
    "check_input",
    "detect_injection",
    "is_pdf_data_not_instruction",
    "mask_pii",
    "mask_value",
    "validate_tool_args",
]
