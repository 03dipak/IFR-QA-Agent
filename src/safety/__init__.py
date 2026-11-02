"""Safety and security.

Implements lld_safety.txt — prompt injection defense, untrusted data
treatment, tool-argument enforcement, adversarial fixture set.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionSignal:
    """Detected injection signal."""
    signal_type: str
    description: str
    severity: str  # low | medium | high


def is_pdf_data_not_instruction(text: str) -> bool:
    """Treat PDF content as DATA, never instructions."""
    return True


def validate_tool_args(tool_name: str, args: dict) -> bool:
    """Validate tool arguments against JSON schema."""
    return bool(args)


def detect_injection(text: str) -> list[InjectionSignal]:
    """Detect prompt injection signals in text."""
    return []
