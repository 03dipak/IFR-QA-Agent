"""Prompt templates for the reflection loop (small-model friendly: short, explicit)."""

from __future__ import annotations

GENERATOR_SYSTEM = (
    "You are the Generator. Answer the question concisely and factually. "
    "If you are not sure, say so in one sentence."
)

CRITIC_SYSTEM = (
    "You are the Critic. Judge the draft answer to the question.\n"
    'Reply with ONLY this JSON, no other text:\n'
    '{"score": <1-5>, "issues": ["<short issue>"], "verdict": "pass" or "revise"}\n'
    "Rules: score 5 = correct and complete; verdict is \"pass\" only when score >= 4.\n"
    'Example: {"score": 3, "issues": ["missing the fiscal-year number"], "verdict": "revise"}'
)

REVISER_SYSTEM = (
    "You are the Reviser. Improve the draft answer by fixing every critic issue. "
    "Keep it concise and do not add new claims."
)


def generator_user(query: str) -> str:
    return f"Question: {query}\n\nAnswer:"


def critic_user(query: str, draft: str) -> str:
    return f"Question: {query}\n\nDraft answer:\n{draft}\n\nRespond with ONLY the JSON object."


def reviser_user(query: str, draft: str, issues: list[str]) -> str:
    bullets = "\n".join(f"- {issue}" for issue in issues) or "- none"
    return (
        f"Question: {query}\n\nDraft:\n{draft}\n\nCritic issues:\n{bullets}\n\n"
        "Return the improved answer only."
    )
