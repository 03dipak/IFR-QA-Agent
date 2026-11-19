"""Prompt builders for the Week-2 prompting strategies."""

from __future__ import annotations

import json

from src.eval.prompts import (
    S1_SYSTEM, S2_SYSTEM, S3_STEP1_SYSTEM, S3_STEP2_SYSTEM, S3_STEP3_SYSTEM,
    render_prompt,
)


def _few_shot_text(rows: list[dict]) -> str:
    lines = ["Few-shot examples (non-gold corpus pages):"]
    for r in rows:
        lines.append(f"Q: {r['question']}")
        lines.append(f"A: {json.dumps(r['answer'], sort_keys=True, separators=(',', ':'))}")
    return "\n".join(lines)


def build_s1_messages(row: dict, *, context: str | None, few_shot_rows: list[dict]) -> list[dict]:
    system = S1_SYSTEM + "\n\n" + _few_shot_text(few_shot_rows)
    user = []
    if context:
        user.append(context)
    user.append(f"Question: {row.get('question_text') or row.get('question') or ''}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user)}]


def build_s2_messages(row: dict, *, context: str | None) -> list[dict]:
    user = []
    if context:
        user.append(context)
    user.append(f"Question: {row.get('question_text') or row.get('question') or ''}")
    return [{"role": "system", "content": S2_SYSTEM},
            {"role": "user", "content": "\n\n".join(user)}]


def build_s3_step1_messages(row: dict, *, context: str | None) -> list[dict]:
    user = []
    if context:
        user.append(context)
    user.append(f"Question: {row.get('question_text') or row.get('question') or ''}")
    return [{"role": "system", "content": S3_STEP1_SYSTEM},
            {"role": "user", "content": "\n\n".join(user)}]


def build_s3_step2_messages(extracted: str) -> list[dict]:
    return [{"role": "system", "content": S3_STEP2_SYSTEM},
            {"role": "user", "content": f"Extracted figures JSON:\n{extracted}"}]


def build_s3_step3_messages(row: dict, *, extracted: str, computed: str) -> list[dict]:
    user = [
        f"Question: {row.get('question_text') or row.get('question') or ''}",
        f"Extracted figures JSON:\n{extracted}",
        f"Computed result JSON:\n{computed}",
    ]
    return [{"role": "system", "content": S3_STEP3_SYSTEM},
            {"role": "user", "content": "\n\n".join(user)}]
