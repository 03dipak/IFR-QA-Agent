"""Answer baseline scoring: units normaliser + exact/tolerance numeric match,
abstain correctness, citation page match, and the single-shot prompt builder.

No network calls happen unless OPENAI_BASE_URL, OPENAI_API_KEY and
OPENAI_MODEL are all set (hosted <=8B OpenAI-compatible endpoint).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

_NUM_RE = re.compile(r"-?\(?\d[\d,]*(?:\.\d+)?\)?")
_MULT = {"crore": 1e7, "crores": 1e7, "lakh": 1e5, "lakhs": 1e5,
         "thousand": 1e3, "million": 1e6, "billion": 1e9}
_CURRENCY = ("₹", "rs.", "rs ", "rs", "inr", "rupee", "rupees")


def to_absolute_rupees(text: str | None) -> float | None:
    """Parse an Indian-formatted money string to absolute rupees.

    Handles ₹, Rs, Rs., INR, parentheses-negative, commas, and the
    crore/lakh scale words. Plain numbers (no scale word) are abs rupees.
    """
    if not text:
        return None
    s = text.strip().lower()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip()
    for cur in _CURRENCY:
        s = s.replace(cur, " ")
    s = s.replace(",", "")
    mult = 1.0
    for word, m in _MULT.items():
        if re.search(rf"\b{word}\b", s):
            mult = m
            s = re.sub(rf"\b{word}\b", " ", s)
            break
    m = _NUM_RE.search(s.replace(" ", ""))
    if not m:
        return None
    try:
        val = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    return -val * mult if neg else val * mult


def numeric_match(gold: str, answer: str, tol_rel: float = 0.01) -> bool:
    g, a = to_absolute_rupees(gold), to_absolute_rupees(answer)
    if g is None or a is None:
        return False
    if g == 0:
        return a == 0
    return abs(a - g) / abs(g) <= tol_rel


def abstain_correct(model_output: str) -> bool:
    return "abstain" in (model_output or "").lower()


_PAGE_RE = re.compile(r"(?:page\s*|p\.?\s*)(\d+)", re.IGNORECASE)


def cited_pages(model_output: str) -> set[int]:
    return {int(m) for m in _PAGE_RE.findall(model_output or "")}


def citation_match(gold_pages: set[int], model_output: str) -> bool:
    return bool(gold_pages & cited_pages(model_output or ""))


def single_shot_prompt(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[chunk {i+1}] document={c['meta']['document_id'][:12]} "
        f"page={c['meta']['page']}\n{c['text'][:1500]}"
        for i, c in enumerate(chunks))
    system = (
        "You answer questions about Indian company filings (standalone or "
        "consolidated, as printed). Use ONLY the provided context. If the "
        "figure is not printed in the context, reply exactly with one word: "
        "ABSTAIN. Otherwise give the figure as printed, with its unit, and "
        "name the page you used, e.g. 'page 12'.")
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def endpoint_configured() -> bool:
    return all(os.environ.get(k) for k in
               ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"))


def call_endpoint(messages: list[dict], timeout: float = 60.0) -> str:
    """POST to an OpenAI-compatible chat-completions endpoint.

    Reads OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL from the environment.
    Raises RuntimeError if unset. Never logs the key.
    """
    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not (base and key and model):
        raise RuntimeError("answer endpoint not configured: set "
                           "OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL")
    url = base.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": 0}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def score_answer(row: dict, model_output: str) -> dict:
    is_abstain_gold = (row.get("proposed_answer") or "").strip().upper() == "ABSTAIN"
    gold_pages = set()
    for p in (row.get("page") or "").split("|"):
        try:
            gold_pages.add(int(p))
        except ValueError:
            pass
    out = {
        "abstain_correct": abstain_correct(model_output) == is_abstain_gold,
        "citation_match": citation_match(gold_pages, model_output),
    }
    if is_abstain_gold:
        out["numeric_match"] = None
    else:
        out["numeric_match"] = numeric_match(row["proposed_answer"], model_output)
    return out
