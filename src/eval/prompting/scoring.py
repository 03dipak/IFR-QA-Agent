"""Deterministic scoring for Week-1 prompt outputs (no LLM judge)."""

from __future__ import annotations

import json
import re

from src.eval.answer_scoring import to_absolute_rupees

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _extract_json(text: str):
    """Return the last JSON object in text, or None."""
    text = text or ""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except (json.JSONDecodeError, TypeError):
            pass
    decoder = json.JSONDecoder()
    found = None
    i = 0
    n = len(text)
    while i < n:
        try:
            obj, end = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                found = obj
            i += max(1, end)
        except (json.JSONDecodeError, TypeError):
            i += 1
    return found



def _decimals(text: str) -> int:
    m = re.search(r"\d+\.(\d+)", text or "")
    return len(m.group(1)) if m else 0


def _scale(text: str) -> float:
    t = (text or "").lower().replace("₹", "rs").replace("rs.", "rs")
    if "crore" in t or re.search(r"\bcrs?\b", t) or re.search(r"\bcr\.?\b", t):
        return 1e7
    if "lakh" in t or re.search(r"\blacs?\b", t):
        return 1e5
    if "million" in t:
        return 1e6
    if "billion" in t:
        return 1e9
    if "thousand" in t:
        return 1e3
    return 1.0


def parse_prediction(output: str) -> dict:
    output = output or ""
    low = output.lower()
    obj = _extract_json(output)
    if isinstance(obj, dict):
        if obj.get("abstain") is True or str(obj.get("answer_value", "")).lower() == "abstain":
            return {"abstain": True, "answer_kind": "abstain", "values": [], "units": [], "pages": [], "parse_ok": True}
    if "abstain" in low and obj is None:
        return {"abstain": True, "answer_kind": "abstain", "values": [], "units": [], "pages": [], "parse_ok": True}

    obj = _extract_json(output)
    if isinstance(obj, dict):
        vals = []
        for key in ("answer_value", "answer", "value"):
            v = obj.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
            elif isinstance(v, str):
                m = _NUM_RE.search(v.replace(",", ""))
                if m:
                    vals.append(float(m.group(0).replace(",", "")))
        unit = str(obj.get("unit") or obj.get("unit_raw") or "")
        pages = []
        for c in obj.get("citations") or []:
            if isinstance(c, dict) and isinstance(c.get("page"), int):
                pages.append(c["page"])
        if isinstance(obj.get("page"), int):
            pages.append(obj["page"])
        answer_kind = obj.get("answer_kind")
        if answer_kind not in {"numeric", "entity", "categorical", "text"}:
            answer_kind = "numeric" if vals else ("entity" if (obj.get("answer_value") not in (None, "")) else "text")
        return {"abstain": False, "answer_kind": answer_kind, "values": vals, "units": [unit],
                "pages": pages, "parse_ok": True if (vals or obj.get("answer_value") not in (None, "")) else False}

    # Plain baseline format: answer: ... | page: n
    pages = [int(m) for m in re.findall(r"page\s*[:=]?\s*(\d+)", output, re.I)]
    answer_part = re.sub(r"\bpage\s*[:=]?\s*\d+(?:\s*,\s*\d+)*", "", output, flags=re.I)
    unit = ""
    for word in ("crore", "lakh", "million", "billion", "percent", "%"):
        if re.search(rf"\b{re.escape(word)}\b", answer_part, re.I):
            unit = word
            break
    vals = [float(m.replace(",", "")) for m in _NUM_RE.findall(answer_part.replace(",", ""))]
    answer_kind = "numeric" if vals else ("entity" if answer_part.strip() else "text")
    return {"abstain": False, "answer_kind": answer_kind, "values": vals, "units": [unit],
            "pages": pages, "parse_ok": bool(vals or answer_part.strip())}


def _expected_numbers(row: dict) -> list[float]:
    nums: list[float] = []
    if isinstance(row.get("expected_value"), (int, float)):
        nums.append(float(row["expected_value"]))
    text = row.get("expected_answer") or row.get("answer_text") or ""
    for m in _NUM_RE.findall(str(text).replace(",", "")):
        try:
            nums.append(float(m))
        except ValueError:
            pass
    # De-duplicate while preserving order.
    out: list[float] = []
    for n in nums:
        if not out or abs(n - out[-1]) > 1e-9:
            out.append(n)
    return out[:5]


def gold_answer_kind(row: dict) -> str:
    if row.get("expected_behaviour") != "answer":
        return "abstain"
    q = (row.get("question_text") or row.get("question") or "").lower()
    expected = str(row.get("expected_answer") or row.get("answer_text") or "")
    if "neutral" in expected.lower():
        return "categorical"
    if any(w in q for w in ("which", "who reports", "who had", "who has", "larger for", "higher for")):
        return "entity"
    if row.get("expected_value") is None and not _expected_numbers(row):
        return "text"
    return "numeric"


def _normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def expected_entity(row: dict) -> str:
    expected = str(row.get("expected_answer") or row.get("answer_text") or "")
    if "neutral" in expected.lower():
        return "neutral"
    pairs = []
    for part in expected.split(";"):
        if "=" in part:
            label, value = part.split("=", 1)
            try:
                pairs.append((label.strip(), float(value.replace(",", ""))))
            except ValueError:
                pass
    if pairs:
        return max(pairs, key=lambda x: x[1])[0]
    return expected.strip()


def _entity_match(pred: str, expected: str) -> bool:
    p = _normalize_label(pred)
    e = _normalize_label(expected)
    if not p or not e:
        return False
    if e in p or p in e:
        return True
    return set(e.split()).issubset(set(p.split()))


def normalize_answer_value(output: str) -> str:
    """Canonical value used for self-consistency majority voting."""
    output = output or ""
    pred = parse_prediction(output)
    if pred.get("abstain"):
        return "ABSTAIN"
    if pred.get("values"):
        return repr(float(pred["values"][0]))
    # JSON answer_value may be a string label.
    try:
        obj = _extract_json(output)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        value = obj.get("answer_value")
        if isinstance(value, str) and value.strip():
            return _normalize_label(value)
    # Plain non-numeric answer: normalise the answer text before the page marker.
    text = re.sub(r"\bpage\s*[:=]?\s*\d+(?:\s*,\s*\d+)*", "", output, flags=re.I)
    text = re.sub(r"^.*?answer\s*[:=]\s*", "", text, flags=re.I | re.S)
    text = text.split("|", 1)[0].strip()
    return _normalize_label(text) if text else ""


def score_prediction(row: dict, output: str, context_pages: set[tuple[str, int]]) -> dict:
    gold_abstain = row.get("expected_behaviour") == "abstain"
    gold_kind = row.get("answer_kind") or gold_answer_kind(row)
    pred = parse_prediction(output)
    pred_abstain = bool(pred["abstain"])
    false_answer = bool(gold_abstain and not pred_abstain)
    false_abstention = bool((not gold_abstain) and pred_abstain)
    abstain_correct = (gold_abstain == pred_abstain)

    expected_numbers = [] if gold_abstain else _expected_numbers(row)
    numeric_match = None
    entity_match = None
    if gold_abstain or pred_abstain:
        pass
    elif gold_kind == "numeric" and expected_numbers and pred["values"]:
        expected_scale = _scale(row.get("expected_unit_raw") or row.get("expected_unit") or row.get("expected_answer") or "")
        decimals = _decimals(str(row.get("expected_answer") or row.get("expected_value") or ""))
        output_text = " ".join(pred.get("units", []) + [output])
        output_scale = _scale(output_text)
        if output_scale == 1.0 and expected_scale != 1.0:
            # No explicit output unit; treat value as in expected unit.
            output_scale = expected_scale
        expected_canonical = [n * expected_scale for n in expected_numbers]
        output_canonical = [n * output_scale for n in pred["values"]]
        tol = 0.5 * (10 ** (-decimals)) * expected_scale
        if tol == 0:
            tol = 1e-6 * (abs(expected_canonical[0]) if expected_canonical else 1.0)
        # Every expected number must be matched by some output number.
        used: set[int] = set()
        ok = True
        for exp in expected_canonical:
            matches = [i for i, out in enumerate(output_canonical)
                       if i not in used and abs(out - exp) <= tol]
            if not matches:
                ok = False
                break
            used.add(matches[0])
        numeric_match = bool(ok)
        row.setdefault("_scoring_debug", {})["expected_scale"] = expected_scale
        row.setdefault("_scoring_debug", {})["output_scale"] = output_scale

    if (not gold_abstain and not pred_abstain and gold_kind in {"entity", "categorical", "text"}):
        entity_match = _entity_match(" ".join([str(pred.get("answer_kind", ""))] + [str(v) for v in pred["values"]] + [output]),
                                     expected_entity(row))

    cited_pages = set(pred["pages"])
    citation_valid = None
    if not pred_abstain and not gold_abstain:
        context_page_nums = {p for (_, p) in context_pages}
        citation_valid = bool(cited_pages) and cited_pages.issubset(context_page_nums)

    return {
        "gold_abstain": gold_abstain,
        "pred_abstain": pred_abstain,
        "abstain_correct": abstain_correct,
        "false_answer": false_answer,
        "false_abstention": false_abstention,
        "gold_answer_kind": gold_kind,
        "pred_answer_kind": pred.get("answer_kind"),
        "numeric_match": numeric_match,
        "entity_match": entity_match,
        "citation_valid": citation_valid,
        "parse_ok": bool(pred["parse_ok"]),
    }
