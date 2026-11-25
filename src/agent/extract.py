from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_LEADING_NUMBERING = re.compile(r"^\s*(?:\([ivx]+\)|\d+[.)])\s*", re.I)
_TRAILING_SUFFIX = re.compile(r"\s*\((?:1\s*\+\s*2\s*\+\s*3|[a-z]|net)\)\s*$", re.I)
_NOTE_REF = re.compile(r"\s+\d+\s*$")
def _label_aliases(label: str) -> tuple[str, ...]:
    if normalize_label(label).startswith("net profit"):
        return ("Profit for the year",)
    return ()


def normalize_label(label: str) -> str:
    text = str(label or "").strip().lower()
    text = _LEADING_NUMBERING.sub("", text)
    while True:
        updated = _TRAILING_SUFFIX.sub("", text)
        if updated == text:
            break
        text = updated
    text = _NOTE_REF.sub("", text)
    text = re.sub(r"[^a-z0-9&]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_number(raw: str) -> str:
    return raw.strip()


def _to_decimal(value: str | None) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _unit(chunk: dict[str, Any]) -> str:
    text = str(chunk.get("text") or "")
    for line in text.splitlines():
        if line.strip().lower().startswith("unit:"):
            value = line.split(":", 1)[1].strip()
            return value or "unit not printed"
    return "unit not printed"


def _meta(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = chunk.get("meta") or {}
    return meta if isinstance(meta, dict) else {}


def _company(chunk: dict[str, Any]) -> str:
    meta = _meta(chunk)
    return str(meta.get("company_id") or meta.get("company") or "").strip()


def _page(chunk: dict[str, Any]) -> int | None:
    value = _meta(chunk).get("page")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _basis(chunk: dict[str, Any]) -> str:
    return str(_meta(chunk).get("reporting_basis") or "").strip().lower()


def _statement_kind(chunk: dict[str, Any]) -> str:
    return str(_meta(chunk).get("statement_kind") or "").strip().lower()


def _period(chunk: dict[str, Any]) -> str:
    return str(_meta(chunk).get("period") or "").strip().lower()


def _value_key(value: object) -> str:
    return str(value or "").replace(",", "").strip()


def _valid_value(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if re.fullmatch(r"\d{1,3}(?:,\d{2})+,\d{3}(?:\.\d+)?", text):
        return False
    number = _to_decimal(text)
    return number is not None and number != 0


def _prefer_meta(
    candidates: list[dict[str, Any]], key: str, wanted: str | None
) -> list[dict[str, Any]]:
    if not wanted:
        return candidates
    token = str(wanted).strip().lower()
    matched = [
        candidate for candidate in candidates
        if str(candidate.get(key) or "").strip().lower() == token
    ]
    return matched or candidates


def _order_chunks(
    chunks: list[dict[str, Any]],
    basis: str | None,
    statement_kind: str | None,
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested_basis = (basis or "").strip().lower()
    requested_kind = (statement_kind or "").strip().lower()
    wanted = {normalize_label(label) for label in labels or [] if normalize_label(label)}
    has_basis_match = requested_basis and any(
        _basis(c) == requested_basis for c in chunks
    )
    has_kind_match = requested_kind and any(
        _statement_kind(c) == requested_kind for c in chunks
    )

    def label_rank(chunk: dict[str, Any]) -> int:
        if not wanted:
            return 1
        for line in str(chunk.get("text") or "").splitlines():
            line_label, _ = _line_label(line.split(":", 1)[0])
            if line_label in wanted:
                return 0
        return 1

    def key(chunk: dict[str, Any]) -> tuple[int, int, int, int, str]:
        cb = _basis(chunk)
        ck = _statement_kind(chunk)
        basis_rank = 0 if requested_basis and cb == requested_basis else 1
        kind_rank = 0 if requested_kind and ck == requested_kind else 1
        unclear_rank = 1 if cb == "unclear" or ck == "unclear" else 0
        return (
            label_rank(chunk),
            basis_rank,
            kind_rank,
            unclear_rank,
            str(chunk.get("chunk_id") or chunk.get("id") or ""),
        )

    filtered = chunks
    if has_basis_match:
        filtered = [c for c in filtered if _basis(c) != "unclear"]
    if has_kind_match:
        filtered = [c for c in filtered if _statement_kind(c) != "unclear"]
    return sorted(filtered, key=key)


def _is_note_reference(token: str) -> bool:
    value = token.strip()
    return bool(
        re.fullmatch(r"\d+(?:\.\d+)?[A-Za-z]{1,3}", value)
        or re.fullmatch(r"\d+\s*\(\s*[a-zA-Z]+\s*\)", value)
        or re.fullmatch(r"\([a-z]+\)", value)
    )


def _line_label(body: str) -> tuple[str, list[str]]:
    text = _LEADING_NUMBERING.sub("", body)
    text = re.sub(r"\((?:1\s*\+\s*2\s*\+\s*3|[a-z]|net)\)", " ", text, flags=re.I)
    match = _NUMBER.search(text)
    if not match:
        return normalize_label(text), []
    label = normalize_label(text[: match.start()])
    numbers = [
        _clean_number(m.group(0))
        for m in _NUMBER.finditer(text[match.start():])
        if not _is_note_reference(m.group(0))
    ]
    return label, numbers


def _iter_lines(text: str) -> list[str]:
    lines = text.splitlines()
    lowered = [line.lower() for line in lines]
    if any("balance as at" in line for line in lowered):
        start = max(i for i, line in enumerate(lowered) if "balance as at" in line)
        lines = lines[start + 1:]
    return lines


def _balance_face_chunk(chunk: dict[str, Any]) -> bool:
    faces = {
        "total current assets",
        "total equity and liabilities",
        "total current liabilities",
    }
    return any(
        normalize_label(line.split(":", 1)[0]) in faces
        for line in _iter_lines(str(chunk.get("text") or ""))
    )


def _candidates_for(
    chunks: list[dict[str, Any]],
    company: str,
    label: str,
    *,
    include_multi_year: bool = False,
) -> list[dict[str, Any]]:
    wanted = normalize_label(label)
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        if _company(chunk).lower() != company.lower():
            continue
        text = str(chunk.get("text") or "")
        equity_statement = "balance as at" in text.lower()
        chunk_basis = _basis(chunk)
        chunk_kind = _statement_kind(chunk)
        chunk_period = _period(chunk)
        chosen: dict[str, Any] | None = None
        for line in _iter_lines(text):
            body = line.split(":", 1)[0]
            line_label, numbers = _line_label(body)
            if line_label != wanted or not numbers:
                continue
            equity_profit = equity_statement and wanted == "profit for the year"
            multi_year = len(numbers) >= 3 and not equity_profit
            if multi_year and not include_multi_year:
                continue
            if equity_profit:
                current = numbers[-1]
                prior = None
            elif multi_year:
                current = numbers[0]
                prior = numbers[1] if len(numbers) > 2 else None
            else:
                current = numbers[-2] if len(numbers) >= 2 else numbers[0]
                prior = numbers[-1] if len(numbers) >= 2 else None
            if not _valid_value(current):
                continue
            chosen = {
                "company_id": company,
                "label": label,
                "value": current,
                "prior": prior,
                "unit": _unit(chunk),
                "chunk_id": str(chunk.get("chunk_id") or chunk.get("id") or ""),
                "page": _page(chunk),
                "two_column": True if equity_profit else len(numbers) == 2,
                "equity_statement": equity_profit,
                "face_balance_sheet": _balance_face_chunk(chunk),
                "multi_year_row": multi_year,
                "period": chunk_period,
                "reporting_basis": chunk_basis,
                "statement_kind": chunk_kind,
            }
        if chosen is not None:
            out.append(chosen)
    return out


def _filter_candidates(
    candidates: list[dict[str, Any]],
    period: str | None,
    basis: str | None,
    statement_kind: str | None,
) -> list[dict[str, Any]]:
    pool = _prefer_meta(candidates, "period", period)
    pool = _prefer_meta(pool, "reporting_basis", basis)
    pool = _prefer_meta(pool, "statement_kind", statement_kind)
    two_column = [candidate for candidate in pool if candidate["two_column"]]
    return two_column or pool


def _choose_candidate(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = _value_key(candidate.get("value"))
        counts[key] = counts.get(key, 0) + 1
    best = max(counts.values())
    leaders = {key for key, count in counts.items() if count == best}
    ranked = [
        candidate for candidate in candidates
        if _value_key(candidate.get("value")) in leaders
    ]
    face = [candidate for candidate in ranked if candidate.get("face_balance_sheet")]
    return dict((face or ranked)[0]), leaders


def extract_figures(
    requested: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    basis: str | None = None,
    statement_kind: str | None = None,
    period: str | None = None,
) -> dict[str, Any]:
    labels = [str(item.get("label") or "") for item in requested]
    ordered = _order_chunks(chunks, basis, statement_kind, labels)
    figures: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    conflicts: list[dict[str, Any]] = []
    for request in requested:
        company = str(request.get("company_id") or "").strip()
        label = str(request.get("label") or "").strip()
        if not company or not label:
            continue
        candidates = _filter_candidates(
            _candidates_for(ordered, company, label), period, basis, statement_kind
        )
        label_equivalence = False
        if not candidates and normalize_label(label) == "total assets":
            candidates = _filter_candidates(
                _candidates_for(ordered, company, "Total equity and liabilities"),
                period,
                basis,
                statement_kind,
            )
            label_equivalence = bool(candidates)
        if not candidates:
            for alias in _label_aliases(label):
                candidates = _filter_candidates(
                    _candidates_for(ordered, company, alias), period, basis, statement_kind
                )
                if candidates:
                    label_equivalence = True
                    break
        if not candidates:
            candidates = _filter_candidates(
                _candidates_for(ordered, company, label, include_multi_year=True),
                period,
                basis,
                statement_kind,
            )
        if not candidates:
            missing.append({"company_id": company, "label": label})
            continue
        chosen, leaders = _choose_candidate(candidates)
        chosen["label"] = label
        if len(leaders) > 1:
            conflicts.append({
                "company_id": company,
                "label": label,
                "values": sorted(leaders),
                "chunk_ids": [c["chunk_id"] for c in candidates],
            })
        if label_equivalence:
            chosen["label_equivalence"] = True
        figures.append(chosen)
    return {
        "figures": figures,
        "missing": missing,
        "conflicts": conflicts,
        "insufficient_evidence": bool(missing or conflicts),
    }


def _value_map(figures: list[dict[str, Any]]) -> list[tuple[str, Decimal, dict[str, Any]]]:
    out: list[tuple[str, Decimal, dict[str, Any]]] = []
    for fig in figures:
        dec = _to_decimal(str(fig.get("value")))
        if dec is None:
            continue
        out.append((str(fig.get("company_id") or ""), dec, fig))
    return out


def _diff_text(diff: Decimal) -> str:
    raw = format(diff, ",f")
    if "." in raw:
        raw = raw.rstrip("0").rstrip(".")
    return raw


def _asked_unit(query: str) -> str:
    match = re.search(
        r"(?:rs\.?|inr|₹)\s*(?:in\s*)?(crore|crores|million|millions|lakh|lakhs|billion)\b",
        query, re.I,
    )
    return f"Rs. in {match.group(1).lower().rstrip('s')}" if match else ""


def _display_unit(unit: str, query: str) -> str:
    if unit != "unit not printed":
        return unit
    asked = _asked_unit(query)
    return f"{asked} (as asked; unit not confirmed on the page)" if asked else unit


def build_answer(
    requested: list[dict[str, str]],
    extracted: dict[str, Any],
    query: str,
) -> dict[str, Any]:
    figures = extracted.get("figures") or []
    missing = extracted.get("missing") or []
    conflicts = extracted.get("conflicts") or []
    insufficient = bool(extracted.get("insufficient_evidence"))
    if conflicts:
        parts = []
        for conflict in conflicts:
            parts.append(
                f"Conflicting {conflict['label']} for {conflict['company_id']}: "
                f"{', '.join(conflict['values'])}"
            )
        return {
            "answer": "Conflict: " + "; ".join(parts),
            "citations": [c for c in (conflict["chunk_ids"] for conflict in conflicts) for c in c],
            "insufficient_evidence": True,
            "figures": figures,
            "conflict": True,
        }
    if missing or not figures:
        parts = []
        for fig in figures:
            unit = _display_unit(str(fig.get("unit") or "unit not printed"), query)
            parts.append(
                f"{fig.get('company_id')} {fig.get('label')}: "
                f"{fig.get('value')} ({unit})"
            )
        for item in missing:
            parts.append(f"{item['company_id']}: label not found ({item['label']})")
        if not parts:
            parts.append("label not found (requested figures)")
        return {
            "answer": " ".join(parts),
            "citations": [str(f.get("chunk_id") or "") for f in figures if f.get("chunk_id")],
            "insufficient_evidence": True,
            "figures": figures,
        }

    labels = {str(fig.get("label") or "") for fig in figures}
    lower = query.lower()
    compare = any(
        token in lower
        for token in ("compare", "larger", "higher", "more", "difference", "by how much")
    )
    computed: list[str] = []
    if len(labels) == 1 and len(figures) >= 2 and compare:
        values = _value_map(figures)
        values.sort(key=lambda item: item[1], reverse=True)
        winner_company, winner_value, winner = values[0]
        loser_company, loser_value, _ = values[1]
        diff = winner_value - loser_value
        diff_text = _diff_text(diff)
        computed.append(diff_text)
        label = str(winner.get("label") or "figure")
        loser = next(f for c, _, f in values if c == loser_company)
        winner_unit = _display_unit(str(winner.get("unit") or "unit not printed"), query)
        loser_unit = _display_unit(str(loser.get("unit") or "unit not printed"), query)
        unit = winner_unit if winner_unit != "unit not printed" else loser_unit
        answer = (
            f"{winner_company} reports higher {label}: "
            f"{winner.get('value')} vs {loser_company} {loser.get('value')}; "
            f"difference {diff_text} ({unit})."
        )
    else:
        parts = []
        for fig in figures:
            unit = _display_unit(str(fig.get("unit") or "unit not printed"), query)
            prior = fig.get("prior")
            prior_text = f", prior {prior}" if prior not in (None, "") else ""
            parts.append(
                f"{fig.get('company_id')} {fig.get('label')}: "
                f"{fig.get('value')} ({unit}){prior_text}"
            )
        answer = "; ".join(parts)
    return {
        "answer": answer,
        "citations": [str(f.get("chunk_id") or "") for f in figures if f.get("chunk_id")],
        "insufficient_evidence": insufficient,
        "figures": [
            {
                **fig,
                "period_column": "current",
            }
            for fig in figures
        ],
        "computed_values": computed,
    }
