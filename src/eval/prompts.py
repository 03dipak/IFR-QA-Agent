"""Versioned baseline prompt set for B0/B1/B2 smoke tests.

Few-shot exemplars are supplied by the caller and must come from non-gold
rows (dev_set.jsonl). The prompt-set hash covers the templates and format,
never the question or exemplar values.
"""

from __future__ import annotations

import hashlib
import json

PROMPT_SET_VERSION = "prompt_baselines.v2"

FEW_SHOT_HEADER = "Few-shot examples (non-gold):"
S1_SYSTEM = ("You are a precise financial-results answerer. Use only the provided oracle context. "
            "Return exactly one JSON object with keys: answer_kind, answer_value, unit, period, basis, "
            "citations, abstain, reason. answer_kind is one of numeric, entity, categorical, text. "
            "citations is a list of {document_id, page}. If abstain is true, answer_value must be null. "
            "Do not include markdown or prose outside the JSON.")
S2_SYSTEM = ("You are a precise financial-results answerer. Use only the provided oracle context. "
             "Reason step by step in plain text, then end with exactly one JSON line containing keys: "
             "answer_kind, answer_value, unit, period, basis, citations, abstain, reason. "
             "The final JSON line must be the last line of your response.")
S3_STEP1_SYSTEM = ("You are a precise financial-results extractor. Use only the provided oracle context. "
                   "Return exactly one JSON array of objects with keys: label, value, unit, document_id, page. "
                   "No prose outside the JSON.")
S3_STEP2_SYSTEM = ("You are a precise calculator. Use only the provided extracted figures. "
                   "Return exactly one JSON object with keys: calculation, result, unit. "
                   "No prose outside the JSON.")
S3_STEP3_SYSTEM = ("You are a precise financial-results answerer. Use only the provided figures and computed result. "
                   "Return exactly one JSON object with keys: answer_kind, answer_value, unit, period, basis, "
                   "citations, abstain, reason. No prose outside the JSON.")

SYSTEM = {
    "S1": S1_SYSTEM,
    "S2": S2_SYSTEM,
    "S3": S3_STEP3_SYSTEM,
    "B0": ("You are a precise financial-results answerer. Closed-book: answer only "
           "from the question and the few-shot examples; never invent a figure. "
           "Output exactly one line: answer: <value> | page: <n>, or ABSTAIN."),
    "B1": ("You are a precise financial-results answerer. Use only the provided "
           "oracle context pages. Output exactly one line: "
           "answer: <value> | page: <n>, or ABSTAIN."),
    "B2": ("You are a precise financial-results answerer. Use only the provided "
           "retrieved context. Output exactly one line: "
           "answer: <value> | page: <n>, or ABSTAIN."),
}

CONTEXT_HEADERS = {
    "B1": "Oracle context (gold pages, smoke only):",
    "B2-global": "Context (global retrieval):",
    "B2-filtered": "Context (filtered retrieval by company/period/basis):",
    "S1": "Oracle context (gold pages, smoke only):",
    "S2": "Oracle context (gold pages, smoke only):",
    "S3": "Oracle context (gold pages, smoke only):",
}


def prompt_set_hash() -> str:
    from pathlib import Path
    few_shot_bytes = b""
    p = Path("data/eval/prompts/fewshot_v1.jsonl")
    if p.is_file():
        few_shot_bytes = p.read_bytes()
    payload = json.dumps({"version": PROMPT_SET_VERSION, "system": SYSTEM,
                          "s1_system": S1_SYSTEM, "s2_system": S2_SYSTEM,
                          "s3_step1": S3_STEP1_SYSTEM, "s3_step2": S3_STEP2_SYSTEM,
                          "s3_step3": S3_STEP3_SYSTEM,
                          "few_shot_header": FEW_SHOT_HEADER,
                          "context_headers": CONTEXT_HEADERS,
                          "few_shot_sha256": hashlib.sha256(few_shot_bytes).hexdigest()},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _few_shot_text(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        q = r.get("question_text") or r.get("question") or ""
        a = r.get("expected_answer") or r.get("answer_text") or ""
        page = r.get("pdf_page_index") or r.get("evidence", [{}])[0].get("pdf_page", "?")
        lines.append(f"Q: {q}\nA: answer: {a} | page: {page}")
    return FEW_SHOT_HEADER + "\n" + "\n\n".join(lines)


def build_messages(baseline: str, row: dict, *, context: str | None = None,
                   few_shot_rows: list[dict] | None = None,
                   retrieval_variant: str = "global") -> list[dict]:
    system = SYSTEM[baseline]
    if baseline == "B0":
        system = system + "\n\n" + _few_shot_text(few_shot_rows or [])
    elif baseline == "B1":
        system = system + "\n\n" + CONTEXT_HEADERS["B1"]
    elif baseline == "B2":
        key = f"B2-{retrieval_variant}"
        system = system + "\n\n" + CONTEXT_HEADERS.get(key, CONTEXT_HEADERS["B2-global"])
    user_parts = []
    if context:
        user_parts.append(context)
    user_parts.append(f"Question: {row.get('question_text') or row.get('question') or ''}")
    user_parts.append("If the answer is not present, reply with the single word: ABSTAIN")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)}]


def render_prompt(messages: list[dict]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
