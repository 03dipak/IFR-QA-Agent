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
    if normalized in {"revenue from operations", "total revenue from operations", "revenue"}:
        return (
            "Total revenue from operations",
            "Revenue from operations",
            "Total revenue from contracts with customers",
            "Revenue from contracts with customers",
            "Net sales",
            "Total revenue",
        )
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


_UNIT_MAP = {"mn": "million", "million": "million", "millions": "million",
             "crore": "crore", "crores": "crore", "cr": "crore",
             "lakh": "lakh", "lakhs": "lakh", "lac": "lakh", "lacs": "lakh"}

def _normalize_unit_token(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    lower = value.lower()
    if not re.search(r"(million|crore|cr\b|lakh|lac\b|mn\b)", lower):
        return ""
    # garbled currency symbols (H, I, f, ~, (, {, etc.) are common OCR artifacts
    m = re.search(r"\b(millions?|crores?|cr|lakhs?|lacs?|mn)\b", lower)
    if not m:
        return ""
    return f"Rs. in {_UNIT_MAP.get(m.group(1), m.group(1))}"


def _unit(chunk: dict[str, Any], *, allow_inferred: bool = False) -> str:
    text = str(chunk.get("text") or "")
    meta = _meta(chunk)
    # same-table/chunk unit line or meta
    for source in [text] + [str(meta.get(k) or "") for k in ("unit", "currency_unit", "amount_unit")]:
        for line in source.splitlines():
            candidate = line.split(":", 1)[1].strip() if ":" in line and line.strip().lower().startswith("unit") else line.strip()
            norm = _normalize_unit_token(candidate)
            if norm:
                return norm
    lower = text.lower()
    header_patterns = [
        r"all amounts?\s*(?:are\s*)?in\s*[^\n,]{0,40}?\b(millions?|crores?|cr|lakhs?|lacs?)\b",
        r"[^\w\s]?\s*(?:rs\.?|inr|₹|`|[a-z])\s*(?:in\s*)?\b(millions?|crores?|cr|lakhs?|lacs?)\b",
        r"\bin\s*[^\n,]{0,40}?\b(millions?|crores?|cr|lakhs?|lacs?)\b",
        r"\b(millions?|crores?|lakhs?|lacs?)\b(?=\s*(?:except|unless|$|\)))",
    ]
    for pat in header_patterns:
        m = re.search(pat, lower)
        if m:
            return f"Rs. in {_UNIT_MAP.get(m.group(1), m.group(1))}"
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

    def label_rank(chunk: dict[str, Any]) -> int:
        if not wanted:
            return 1
        for line in str(chunk.get("text") or "").splitlines():
            line_label, _ = _line_label(line.split(":", 1)[0])
            if line_label in wanted:
                return 0
        return 1

    def key(chunk: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
        cb = _basis(chunk) or _basis_from_chunk(chunk)
        ck = _statement_kind(chunk) or _statement_kind_from_header(_statement_header(chunk))
        basis_rank = 0 if (requested_basis and cb == requested_basis) else (0 if (not requested_basis and cb == "consolidated") else 1)
        kind_rank = 0 if requested_kind and ck == requested_kind else 1
        unclear_rank = 1 if cb == "unclear" or ck == "unclear" else 0
        header_rank = 0 if _header_years(str(chunk.get("text") or "").splitlines(), 0) or "year ended" in str(chunk.get("text") or "").lower() else 1
        return (
            label_rank(chunk),
            basis_rank,
            kind_rank,
            unclear_rank,
            header_rank,
            str(chunk.get("chunk_id") or chunk.get("id") or ""),
        )

    return sorted(chunks, key=key)


_YEAR_TOKEN = re.compile(r"\b(20\d{2})\b")
_YEAR_ONLY = re.compile(r"^\s*(?:FY\s*)?(20\d{2})(?:\s*[-–/]\s*(?:20)?\d{2,4})?\s*$", re.I)


def _is_note_reference(token: str) -> bool:
    value = token.strip()
    return bool(
        re.fullmatch(r"\d+(?:\.\d+)?[A-Za-z]{1,3}", value)
        or re.fullmatch(r"\d+\s*\(\s*[a-zA-Z]+\s*\)", value)
        or re.fullmatch(r"\([a-z]+\)", value)
    )


def _document_id(chunk: dict[str, Any]) -> str:
    meta = _meta(chunk)
    doc_id = str(meta.get("document_id") or "").strip()
    if doc_id:
        return doc_id
    chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
    return chunk_id.split(":", 1)[0].strip()


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


def _header_years(lines: list[str], row_index: int) -> list[int]:
    def _years_from_line(line: str) -> list[int]:
        years: list[int] = []
        for match in _YEAR_TOKEN.finditer(str(line or "")):
            year = int(match.group(1))
            if year not in years:
                years.append(year)
        return years

    for idx in range(row_index, max(-1, row_index - 12), -1):
        years = _years_from_line(lines[idx])
        if len(years) >= 2:
            return years
    # header split across lines (e.g. Year ended / March 31, 2025 / Year ended / March 31, 2024)
    for window_size in (2, 3, 4, 5, 6, 8):
        start = max(0, row_index - 1)
        for s in range(start, max(-1, row_index - 30), -1):
            window_lines = [str(lines[i] or "") for i in range(s, min(len(lines), s + window_size))]
            window = " ".join(window_lines)
            low = window.lower()
            if "respectively" in low:
                continue
            years = _years_from_line(window)
            if len(years) >= 2 and ("year ended" in low or "for the year" in low or "as at" in low):
                return years
    window = " ".join(str(lines[i] or "") for i in range(max(0, row_index - 6), row_index + 1))
    years = _years_from_line(window)
    return years if len(years) >= 2 else []


_DASH = re.compile(r"(?<![\w])[-–—](?![\w])")


def _row_tokens(body: str) -> list[str]:
    tokens: list[tuple[int, str]] = []
    for m in _NUMBER.finditer(body):
        if _is_note_reference(m.group(0)):
            continue
        tokens.append((m.start(), _clean_number(m.group(0))))
    for m in _DASH.finditer(body):
        # treat dash as missing only when surrounded by spaces/table separators
        tokens.append((m.start(), "-"))
    tokens.sort(key=lambda x: x[0])
    return [t for _, t in tokens]


def _looks_like_note_number(token: str, value_tokens: int) -> bool:
    text = str(token or "").strip().replace(",", "")
    if not text:
        return False
    if re.fullmatch(r"\d{1,2}", text):
        try:
            return int(text) <= 60 and value_tokens >= 2
        except ValueError:
            return False
    if re.fullmatch(r"\d+\.\d{1,2}", text):
        try:
            num = Decimal(text)
        except (InvalidOperation, ValueError):
            return False
        return num < 100 and value_tokens >= 2
    return False


def _select_row_values(numbers: list[str], header_years: list[int]) -> list[str]:
    if not numbers:
        return []
    if header_years:
        wanted = len(header_years)
        if len(numbers) > wanted:
            tail = numbers[-wanted:]
            dropped = numbers[:-wanted]
            if all(_looks_like_note_number(tok, wanted) for tok in dropped):
                return tail
            if len(numbers) == wanted + 1 and _looks_like_note_number(numbers[0], wanted):
                return tail
            return tail
        return numbers
    return numbers


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
    for line in text.splitlines():
        l = line.lower()
        if "profit and loss" in l or "income statement" in l or "statement of profit" in l:
            return line.strip()
        if "balance sheet" in l or "statement of financial position" in l:
            return line.strip()
    return ""


_NON_FACE_PL_TOKENS = (
    "summarised", "summarized", "summary", "notes to", "note to",
    "segment", "subsidiary", "joint venture", "associate",
    "breakup", "break-up",
)


def _is_non_face_pl_context(chunk: dict[str, Any], lines: list[str], row_index: int) -> bool:
    text_low = str(chunk.get("text") or "").lower()
    head = " ".join(str(l or "").lower() for l in lines[:12])
    if any(tok in head for tok in ("notes to", "note to")):
        return True
    # inspect local section heading near row
    for idx in range(row_index, max(-1, row_index - 12), -1):
        low = str(lines[idx] or "").lower()
        if any(tok in low for tok in _NON_FACE_PL_TOKENS):
            # allow "Statement of Profit and Loss" face heading alone
            if ("statement of profit and loss" in low or "income statement" in low) and not any(
                t in low for t in ("summarised", "summarized", "summary", "reconciliation")
            ):
                continue
            return True
        if "statement of profit and loss" in low or "income statement" in low or "statement of profit" in low:
            if any(t in low for t in ("summarised", "summarized", "summary")):
                return True
            return False
    return any(tok in text_low[:220] for tok in ("notes to", "note to"))


def _section_kind_at(lines: list[str], row_index: int) -> str | None:
    last_kind: str | None = None
    for idx in range(row_index, -1, -1):
        low = str(lines[idx] or "").lower().strip()
        if "profit and loss" in low or "income statement" in low or "statement of profit" in low:
            return "income_statement"
        if "cash flow" in low or "statement of cash flows" in low:
            return "cash_flow"
        if "balance sheet" in low or "statement of financial position" in low:
            last_kind = "balance_sheet"
            continue
        if last_kind == "balance_sheet":
            if low in {"income", "income:"} or low.startswith("revenue from operations") or low.startswith("total income"):
                return "income_statement"
            if "total equity and liabilities" in low or "total assets" == low:
                return "balance_sheet"
            if any(k in low for k in ("profit before tax", "tax expense", "earnings per equity share", "total expenses")):
                return "income_statement"
    return last_kind


def _basis_from_chunk(chunk: dict[str, Any]) -> str:
    basis = _basis(chunk)
    if basis:
        return basis
    header = str(_statement_header(chunk) or "").lower()
    if "consolidated" in header:
        return "consolidated"
    if "standalone" in header:
        return "standalone"
    text = str(chunk.get("text") or "").lower()
    if any(k in text for k in (
        "consolidated statement of profit", "consolidated statement of financial",
        "consolidated balance sheet", "consolidated cash flow",
        "consolidated financial statements",
    )):
        return "consolidated"
    if any(k in text for k in (
        "standalone statement of profit", "standalone statement of financial",
        "standalone balance sheet", "standalone cash flow",
        "standalone financial statements",
    )):
        return "standalone"
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


def _requested_year(period: str | None) -> int | None:
    if not period:
        return None
    match = re.search(r"(20\d{2})", str(period))
    return int(match.group(1)) if match else None


def _is_pl_statement(chunk: dict[str, Any], section_kind: str | None = None) -> bool:
    sk = _statement_kind(chunk)
    if sk == "balance_sheet":
        return False
    if section_kind == "balance_sheet":
        return False
    header = _statement_header(chunk).lower()
    if header and ("profit and loss" in header or "income statement" in header or "statement of profit" in header):
        return True
    if section_kind == "income_statement":
        return True
    return sk == "income_statement"


def _statement_kind_from_header(header: str) -> str:
    low = str(header or "").lower()
    if "profit and loss" in low or "income statement" in low or "statement of profit" in low:
        return "income_statement"
    if "balance sheet" in low or "statement of financial position" in low:
        return "balance_sheet"
    if "cash flow" in low or "statement of cash flows" in low:
        return "cash_flow"
    return ""


def _candidate_base(chunk: dict[str, Any], company: str, label: str) -> dict[str, Any]:
    meta_period = str(_meta(chunk).get("period") or "").strip()
    header = _statement_header(chunk)
    basis = _basis(chunk) or _basis_from_chunk(chunk)
    kind = _statement_kind(chunk) or _statement_kind_from_header(header)
    return {
        "company_id": company,
        "label": label,
        "unit": _unit(chunk),
        "chunk_id": str(chunk.get("chunk_id") or chunk.get("id") or ""),
        "document_id": _document_id(chunk),
        "page": _page(chunk),
        "meta_period": meta_period,
        "reporting_basis": basis,
        "statement_kind": kind,
        "statement_header": header,
        "face_balance_sheet": _balance_face_chunk(chunk),
    }


def _candidates_for(
    chunks: list[dict[str, Any]],
    company: str,
    label: str,
    *,
    include_multi_year: bool = False,
    require_pl: bool = False,
    period: str | None = None,
) -> list[dict[str, Any]]:
    wanted = normalize_label(label)
    target_year = _requested_year(period)
    out: list[dict[str, Any]] = []

    for chunk in chunks:
        if _company(chunk).lower() != company.lower():
            continue
        text = str(chunk.get("text") or "")
        lines = _iter_lines(text)
        equity_statement = "balance as at" in text.lower()
        base = _candidate_base(chunk, company, label)
        chunk_basis_text = " ".join(str(l).lower() for l in lines[:30])
        has_standalone = "standalone" in chunk_basis_text
        has_consolidated = "consolidated" in chunk_basis_text
        if has_standalone and not has_consolidated:
            base["reporting_basis"] = "standalone"
        elif has_standalone and has_consolidated:
            base["reporting_basis"] = ""
        chunk_has_pl = any(
            k in str(l).lower() for l in lines
            for k in ("profit and loss", "income statement", "statement of profit")
        )
        if require_pl and _statement_kind(chunk) == "balance_sheet" and not chunk_has_pl:
            continue
        for idx, line in enumerate(lines):
            body = line.split(":", 1)[0]
            line_label, raw_numbers = _line_label(body)
            if line_label != wanted or not raw_numbers:
                continue
            section_kind = _section_kind_at(lines, idx)
            if require_pl and not _is_pl_statement(chunk, section_kind):
                if section_kind != "balance_sheet" and (chunk_has_pl or _statement_kind(chunk) == "income_statement"):
                    pass
                else:
                    continue
            if require_pl and _is_non_face_pl_context(chunk, lines, idx):
                continue
            header_years = _header_years(lines, idx)
            numbers = _select_row_values(raw_numbers, header_years)
            equity_profit = equity_statement and wanted == "profit for the year"
            multi_year = len(numbers) >= 3 and not equity_profit
            if multi_year and not header_years and not include_multi_year:
                continue
            header_text = ""
            if header_years:
                for hidx in range(idx, max(-1, idx - 8), -1):
                    hline = str(lines[hidx] or "")
                    if len(set(int(m.group(1)) for m in _YEAR_TOKEN.finditer(hline))) >= 2:
                        header_text = hline.strip()
                        break
                row_tokens = _row_tokens(body)
                if len(row_tokens) == len(header_years):
                    year_values = list(zip(header_years, row_tokens))
                elif len(row_tokens) > len(header_years):
                    dropped = row_tokens[:-len(header_years)]
                    tail = row_tokens[-len(header_years):]
                    if all(_looks_like_note_number(t, len(header_years)) for t in dropped):
                        year_values = list(zip(header_years, tail))
                    else:
                        continue
                else:
                    continue
                value = prior = None
                chosen_year = None
                comparative = False
                if target_year is not None:
                    for pos, (year, val) in enumerate(year_values):
                        if year == target_year:
                            value = val
                            chosen_year = year
                            older = [(y, v) for y, v in year_values if y < year and _valid_value(v)]
                            if older:
                                older.sort(key=lambda x: x[0], reverse=True)
                                prior = older[0][1]
                            comparative = bool(base.get("meta_period")) and str(base.get("meta_period")).upper() != f"FY{year}"
                            break
                if value is None and target_year is None:
                    latest = max(year_values, key=lambda x: x[0])
                    chosen_year, value = latest
                    older = [(y, v) for y, v in year_values if y < chosen_year and _valid_value(v)]
                    if older:
                        older.sort(key=lambda x: x[0], reverse=True)
                        prior = older[0][1]
                if value is None:
                    continue
                if str(value).strip() in {"-", "–", "—"}:
                    continue
                if not _valid_value(value):
                    continue
                if prior and re.search(r"\d", str(prior)) and not _valid_value(prior):
                    prior = None
                current_decimal = _to_decimal(value)
                prior_decimal = _to_decimal(str(prior)) if prior else None
                if (
                    re.fullmatch(r"\d{1,3}", str(value))
                    and current_decimal is not None
                    and prior_decimal is not None
                    and abs(prior_decimal) >= 1000
                    and abs(prior_decimal / current_decimal) >= 50
                ):
                    continue
                candidate = dict(base)
                note_chunk = "notes to" in str(chunk.get("text") or "").lower()[:180]
                candidate.update({
                    "value": value,
                    "prior": prior,
                    "period": f"FY{chosen_year}" if chosen_year else str(period or ""),
                    "column_year": chosen_year,
                    "header_years": header_years,
                    "column_header": header_text or base.get("statement_header"),
                    "two_column": len(year_values) == 2,
                    "equity_statement": equity_profit,
                    "multi_year_row": multi_year,
                    "comparative_column": comparative,
                    "source_period": str(base.get("meta_period") or "").upper() or None,
                    "raw_row": body.strip(),
                    "note_chunk": note_chunk,
                })
                out.append(candidate)
                continue

            # fallback: no header years. For annual FY requests, require explicit header year mapping
            # when the chunk has year headers elsewhere; allow legacy meta-period fallback for headerless chunks.
            if target_year is not None and re.fullmatch(r"FY\d{4}", str(period or "").upper()):
                has_any_header = any(len(_YEAR_TOKEN.findall(str(l or ""))) >= 2 for l in lines)
                if has_any_header:
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
            candidate = dict(base)
            candidate.update({
                "value": current,
                "prior": prior,
                "period": str(base.get("meta_period") or period or "").lower(),
                "two_column": True if equity_profit else len(numbers) == 2,
                "equity_statement": equity_profit,
                "multi_year_row": multi_year,
                "raw_row": body.strip(),
            })
            out.append(candidate)
    return out


def _filter_candidates(
    candidates: list[dict[str, Any]],
    period: str | None,
    basis: str | None,
    statement_kind: str | None,
) -> list[dict[str, Any]]:
    pool = list(candidates)
    wanted_period = str(period or "").strip().lower()
    if wanted_period:
        period_exact = [
            c for c in pool
            if str(c.get("period") or "").strip().lower() == wanted_period
        ]
        if period_exact:
            pool = period_exact
        else:
            if re.fullmatch(r"fy\d{4}", wanted_period):
                legacy = [c for c in pool if c.get("column_year") is None]
                pool = legacy
    if basis:
        wanted_basis = str(basis).strip().lower()
        basis_exact = [c for c in pool if str(c.get("reporting_basis") or "").strip().lower() == wanted_basis]
        if basis_exact:
            pool = basis_exact
    else:
        consolidated = [c for c in pool if str(c.get("reporting_basis") or "").strip().lower() == "consolidated"]
        standalone = [c for c in pool if str(c.get("reporting_basis") or "").strip().lower() == "standalone"]
        if consolidated:
            pool = consolidated
        elif standalone:
            pool = standalone
    if statement_kind:
        wanted_kind = str(statement_kind).strip().lower()
        kind_exact = [c for c in pool if str(c.get("statement_kind") or "").strip().lower() == wanted_kind]
        if kind_exact:
            pool = kind_exact
    else:
        income = [c for c in pool if str(c.get("statement_kind") or "").strip().lower() == "income_statement"]
        if income and any(str(c.get("statement_kind") or "").strip() for c in pool):
            # prefer explicit income statement for revenue-like labels; safe because require_pl already gates
            pool = income
    two_column = [candidate for candidate in pool if candidate.get("two_column")]
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
            _candidates_for(ordered, company, label, require_pl=require_pl, period=request_period),
            request_period, basis, statement_kind,
        )
        label_equivalence = False
        if not candidates and normalize_label(label) == "total assets":
            candidates = _filter_candidates(
                _candidates_for(ordered, company, "Total equity and liabilities", require_pl=False, period=request_period),
                request_period,
                basis,
                statement_kind,
            )
            label_equivalence = bool(candidates)
        if not candidates:
            for alias in _label_aliases(label):
                candidates = _filter_candidates(
                    _candidates_for(ordered, company, alias, require_pl=require_pl, period=request_period),
                    request_period, basis, statement_kind,
                )
                if candidates:
                    label_equivalence = True
                    break
        if not candidates:
            candidates = _filter_candidates(
                _candidates_for(ordered, company, label, include_multi_year=True, require_pl=require_pl, period=request_period),
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
