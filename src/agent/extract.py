from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_GROUPED_NUMBER = re.compile(
    r"^-?(?:\d{1,2}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$"
)
_LEADING_NUMBERING = re.compile(r"^\s*(?:\([ivx]+\)|[ivx]+\s+|\d+[.)]|\d+(?=\s+[A-Za-z]))\s*", re.I)
_TRAILING_SUFFIX = re.compile(r"\s*\((?:1\s*\+\s*2\s*\+\s*3|[a-z]|net)\)\s*$", re.I)
_NOTE_REF = re.compile(r"\s+\d+\s*$")
def _label_aliases(label: str) -> tuple[str, ...]:
    normalized = normalize_label(label)
    if normalized == "revenue from operations":
        return ("Total revenue from operations", "Net sales", "Total revenue")
    if normalized.startswith("net profit"):
        return ("Profit for the year",)
    return ()


def normalize_label(label: str) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"(?<![a-z0-9])([a-z])\s+(?=[a-z]{3,})", r"\1", text)
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


def _unit(chunk: dict[str, Any], *, allow_inferred: bool = False) -> str:
    text = str(chunk.get("text") or "")
    for line in text.splitlines():
        if line.strip().lower().startswith("unit:"):
            value = line.split(":", 1)[1].strip()
            if value and re.search(r"(million|crore|lakh|mn|\bcr\b|lacs?)", value.lower()):
                return value
    meta = _meta(chunk)
    for key in ("unit", "currency_unit", "amount_unit"):
        value = str(meta.get(key) or "").strip()
        if value and re.search(r"(million|crore|lakh|mn|\bcr\b|lacs?)", value.lower()):
            return value
    lower = text.lower()
    header_patterns = [
        r"all amounts?\s*(?:are\s*)?in\s*[^\n,]{0,30}?(million|crore|cr|lakhs?|lacs?)\b",
        r"(?:rs\.?|inr|₹|`|i|c)\s*(?:in\s*)?(million|crore|cr|lakhs?|lacs?)\b",
        r"\bin\s*[^\n,]{0,30}?(million|crore|cr|lakhs?|lacs?)\b",
    ]
    for pat in header_patterns:
        m = re.search(pat, lower)
        if m:
            token = m.group(1)
            mapping = {"mn":"million","million":"million","crore":"crore","cr":"crore","lakh":"lakh","lakhs":"lakh","lac":"lakh","lacs":"lakh"}
            return f"Rs. in {mapping.get(token, token)}"
    if allow_inferred:
        return "unit not stated in source"
    return "unit not stated in source"


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
    if "," in text and not _GROUPED_NUMBER.fullmatch(text):
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


def _statement_header(chunk: dict[str, Any]) -> str:
    text = str(chunk.get("text") or "")
    for line in text.splitlines()[:8]:
        l = line.lower()
        if "profit and loss" in l or "income statement" in l or "statement of profit" in l:
            return line.strip()
    return ""


def _is_pl_statement(chunk: dict[str, Any]) -> bool:
    sk = _statement_kind(chunk)
    if sk == "balance_sheet":
        return False
    header = _statement_header(chunk).lower()
    if header and ("profit and loss" in header or "income statement" in header or "statement of profit" in header):
        return True
    if sk == "income_statement":
        return True
    return False


def _candidates_for(
    chunks: list[dict[str, Any]],
    company: str,
    label: str,
    *,
    include_multi_year: bool = False,
    require_pl: bool = False,
) -> list[dict[str, Any]]:
    wanted = normalize_label(label)
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        if _company(chunk).lower() != company.lower():
            continue
        if require_pl and not _is_pl_statement(chunk):
            continue
        text = str(chunk.get("text") or "")
        equity_statement = "balance as at" in text.lower()
        chunk_basis = _basis(chunk)
        chunk_kind = _statement_kind(chunk)
        chunk_period = _period(chunk)
        statement_header = _statement_header(chunk)
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
            if prior and re.search(r"\d", str(prior)) and not _valid_value(prior):
                continue
            current_decimal = _to_decimal(current)
            prior_decimal = _to_decimal(str(prior)) if prior else None
            if (
                re.fullmatch(r"\d{1,3}", str(current))
                and current_decimal is not None
                and prior_decimal is not None
                and abs(prior_decimal) >= 1000
                and abs(prior_decimal / current_decimal) >= 50
            ):
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
                "statement_header": statement_header,
                "raw_row": body.strip(),
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


def _candidate_group_key(candidate: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(candidate.get("company_id") or "").strip().lower(),
        str(candidate.get("period") or "").strip().lower(),
        str(candidate.get("reporting_basis") or "").strip().lower(),
        str(candidate.get("unit") or "").strip().lower(),
    )


def _period_rank(period: str, wanted: str | None) -> tuple[int, int]:
    if not wanted:
        return (0, 0)
    if period == wanted:
        return (0, 0)
    match = re.search(r"(\d{4})", period)
    wanted_match = re.search(r"(\d{4})", wanted)
    if match and wanted_match:
        year = int(match.group(1))
        asked = int(wanted_match.group(1))
        if year < asked:
            return (1, asked - year)
        return (2, year - asked)
    return (3, 0)


def _choose_candidate_group(
    candidates: list[dict[str, Any]], request_period: str | None
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(_candidate_group_key(candidate), []).append(candidate)
    ranked = sorted(
        groups.values(),
        key=lambda group: (
            min(_period_rank(str(c.get("period") or ""), request_period) for c in group),
            -len(group),
        ),
    )
    return ranked[0]


def _conflict_details(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "value": c.get("value"),
            "chunk_id": c.get("chunk_id"),
            "page": c.get("page"),
            "period": c.get("period"),
            "reporting_basis": c.get("reporting_basis"),
            "unit": c.get("unit"),
        }
        for c in candidates
    ]


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
        request_period = str(request.get("period") or period or "").strip() or None
        if not company or not label:
            continue
        require_pl = normalize_label(label) in {"total expenses", "revenue from operations", "total revenue", "total income", "net sales", "profit for the year", "profit after tax", "net profit"}
        candidates = _filter_candidates(
            _candidates_for(ordered, company, label, require_pl=require_pl), request_period, basis, statement_kind
        )
        label_equivalence = False
        if not candidates and normalize_label(label) == "total assets":
            candidates = _filter_candidates(
                _candidates_for(ordered, company, "Total equity and liabilities", require_pl=False),
                request_period,
                basis,
                statement_kind,
            )
            label_equivalence = bool(candidates)
        if not candidates:
            for alias in _label_aliases(label):
                candidates = _filter_candidates(
                    _candidates_for(ordered, company, alias, require_pl=require_pl), request_period, basis, statement_kind
                )
                if candidates:
                    label_equivalence = True
                    break
        if not candidates:
            candidates = _filter_candidates(
                _candidates_for(ordered, company, label, include_multi_year=True, require_pl=require_pl),
                request_period,
                basis,
                statement_kind,
            )
        if not candidates:
            missing.append({"company_id": company, "label": label})
            continue
        group = _choose_candidate_group(candidates, request_period)
        chosen, leaders = _choose_candidate(group)
        display_label = f"{label} ({request_period})" if request_period else label
        chosen["label"] = display_label
        if len(leaders) > 1:
            conflicts.append({
                "company_id": company,
                "label": display_label,
                "values": sorted(leaders),
                "chunk_ids": [c["chunk_id"] for c in group],
                "group": _candidate_group_key(chosen),
                "details": _conflict_details(group),
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
    if unit != "unit not stated in source":
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
            unit = _display_unit(str(fig.get("unit") or "unit not stated in source"), query)
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
        winner_unit = _display_unit(str(winner.get("unit") or "unit not stated in source"), query)
        loser_unit = _display_unit(str(loser.get("unit") or "unit not stated in source"), query)
        unit = winner_unit if winner_unit != "unit not stated in source" else loser_unit
        answer = (
            f"{winner_company} reports higher {label}: "
            f"{winner.get('value')} vs {loser_company} {loser.get('value')}; "
            f"difference {diff_text} ({unit})."
        )
    else:
        parts = []
        for fig in figures:
            unit = _display_unit(str(fig.get("unit") or "unit not stated in source"), query)
            prior = fig.get("prior")
            prior_text = f", prior {prior}" if prior not in (None, "") else ""
            parts.append(
                f"{fig.get('company_id')} {fig.get('label')}: "
                f"{fig.get('value')} ({unit}){prior_text}"
            )
        if len(figures) == 1 and any(t in lower for t in ("growth", "%", "percent", "percentage", "change")):
            fig = figures[0]
            growth_value = fig.get("growth_percent")
            if growth_value is None:
                cur = _to_decimal(str(fig.get("value")))
                prior = _to_decimal(str(fig.get("prior")))
                if cur is not None and prior is not None and prior != 0:
                    growth_value = (cur - prior) / prior * Decimal("100")
            if growth_value is not None:
                growth = Decimal(str(growth_value)).quantize(Decimal("0.01"))
                growth_text = format(growth, "f").rstrip("0").rstrip(".")
                answer = "; ".join(parts) + f"; growth {growth_text}% (calculated)"
                computed.append(growth_text)
            else:
                answer = "; ".join(parts)
        else:
            answer = "; ".join(parts)
    citation_ids: list[str] = []
    for fig in figures:
        for key in ("chunk_id", "unit_chunk_id"):
            cid = str(fig.get(key) or "")
            if cid and cid not in citation_ids:
                citation_ids.append(cid)
    return {
        "answer": answer,
        "citations": citation_ids,
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
