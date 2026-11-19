"""Arithmetic checks on statement tables inside proposed spans.

Implements lld_tables.txt QA arithmetic: balance-sheet balance, subtotal sums
and cash-flow opening/closing reconciliation. Only table text inside a span in
the proposed statement map is read. Results are PASS, FAIL or NOT_CHECKABLE.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

ABS_TOL = Decimal("0.01")
REL_TOL = Decimal("0.00001")

_NUMBER = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")


@dataclass
class ParsedRow:
    label: str
    values: tuple[Decimal, ...] = ()


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | NOT_CHECKABLE
    detail: str = ""


@dataclass
class StatementResult:
    company: str
    checksum: str
    basis: str
    statement: str
    start_pdf_page: int
    end_pdf_page: int
    status: str
    checks: list[Check]


def _decimal(token: str) -> Decimal | None:
    token = token.strip()
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").replace(",", "")
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    return -value if negative else value


def parse_rows(text: str) -> list[ParsedRow]:
    rows: list[ParsedRow] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", " ", raw_line)
        line = re.sub(r"(?<!\S)[-\u2013\u2014](?!\S)", "0", line)
        values = [value for value in (_decimal(m) for m in _NUMBER.findall(line))
                  if value is not None]
        if not values:
            continue
        label = _NUMBER.sub(" ", line)
        label = " ".join(re.sub(r"[^A-Za-z&() ]+", " ", label).split())
        if not label:
            continue
        rows.append(ParsedRow(label=label, values=tuple(values[-2:])))
    return rows


def _norm(label: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9& ]+", " ", label.lower()).split())


def _find(rows: list[ParsedRow], pattern: str,
          exclude: str | None = None,
          min_values: int = 0) -> ParsedRow | None:
    regex = re.compile(pattern, re.I)
    skip = re.compile(exclude, re.I) if exclude else None
    fallback: ParsedRow | None = None
    for row in rows:
        label = _norm(row.label)
        if regex.search(label) and not (skip and skip.search(label)):
            if len(row.values) >= min_values:
                return row
            if fallback is None:
                fallback = row
    return fallback


def _approx(a: Decimal, b: Decimal) -> bool:
    tolerance = max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))
    return abs(a - b) <= tolerance


def _columns(*rows: ParsedRow | None) -> list[int]:
    lengths = [len(row.values) for row in rows if row is not None and row.values]
    return list(range(min(lengths))) if lengths else []


def _subtotal_check(name: str, total: ParsedRow | None,
                    parts: Sequence[ParsedRow | None]) -> Check:
    present_parts = [part for part in parts if part is not None and part.values]
    if total is None or not total.values or not present_parts:
        return Check(name, "NOT_CHECKABLE", "missing total or parts")
    for column in _columns(total, *present_parts):
        expected = sum((part.values[column] for part in present_parts), Decimal("0"))
        if not _approx(total.values[column], expected):
            return Check(name, "FAIL",
                         f"column {column + 1}: {total.values[column]} != {expected}")
    return Check(name, "PASS")


def _equality_check(name: str, left: ParsedRow | None,
                    right: ParsedRow | None) -> Check:
    if left is None or right is None or not left.values or not right.values:
        return Check(name, "NOT_CHECKABLE", "missing row")
    for column in _columns(left, right):
        if not _approx(left.values[column], right.values[column]):
            return Check(name, "FAIL",
                         f"column {column + 1}: {left.values[column]} != {right.values[column]}")
    return Check(name, "PASS")


def _sum_check(name: str, total: ParsedRow | None,
               parts: Sequence[ParsedRow | None]) -> Check:
    present = [part for part in parts if part is not None and part.values]
    if total is None or not total.values or not present:
        return Check(name, "NOT_CHECKABLE", "missing total or parts")
    for column in _columns(total, *present):
        expected = sum((part.values[column] for part in present), Decimal("0"))
        if not _approx(total.values[column], expected):
            return Check(name, "FAIL",
                         f"column {column + 1}: {total.values[column]} != {expected}")
    return Check(name, "PASS")


def _overall(checks: list[Check]) -> str:
    if any(check.status == "FAIL" for check in checks):
        return "FAIL"
    if any(check.status == "PASS" for check in checks):
        return "PASS"
    return "NOT_CHECKABLE"


def check_balance_sheet(rows: list[ParsedRow]) -> list[Check]:
    total_assets = _find(rows, r"\btotal assets?\b")
    total_el = _find(rows, r"\btotal (equity and liabilities|liabilities and equity)\b")
    total_equity = _find(rows, r"\btotal equity\b", exclude=r"liabilities|attributable")
    share_capital = _find(rows, r"\bequity share capital\b")
    other_equity = _find(rows, r"\bother equity\b")
    nci = _find(rows, r"\bnon controlling interests?\b")
    held_for_sale = _find(rows, r"\bassets held for sale\b|\bassets classified as held for sale\b")
    nca = _find(rows, r"\btotal non current assets?\b")
    ca = _find(rows, r"\btotal current assets?\b")
    ncl = _find(rows, r"\btotal non current liabilities\b")
    cl = _find(rows, r"\btotal current liabilities\b")
    total_liabilities = _find(rows, r"\btotal liabilities\b")
    held_sale_liabilities = _find(
        rows, r"\bliabilities associated with.*held for sale\b")

    checks = [
        _equality_check("assets_equals_equity_and_liabilities", total_assets, total_el),
        _sum_check("total_equity_equals_parts", total_equity,
                   [share_capital, other_equity, nci]),
        _sum_check("assets_equal_current_and_non_current", total_assets,
                   [nca, ca, held_for_sale]),
    ]
    if total_liabilities is not None:
        checks.append(_sum_check("equity_and_liabilities_equal_parts",
                                 total_el, [total_equity, total_liabilities]))
    if total_liabilities is None:
        if ncl is not None or cl is not None:
            checks.append(Check("liabilities_equal_current_and_non_current",
                                "NOT_CHECKABLE", "missing total liabilities"))
    elif ncl is not None or cl is not None:
        checks.append(_sum_check("liabilities_equal_current_and_non_current",
                                 total_liabilities,
                                 [ncl, cl, held_sale_liabilities]))
    return checks


def _section_items(rows: list[ParsedRow], start_pattern: str,
                   stop_pattern: str) -> list[ParsedRow]:
    out: list[ParsedRow] = []
    active = False
    for row in rows:
        label = _norm(row.label)
        if re.search(stop_pattern, label):
            break
        if re.search(start_pattern, label):
            active = True
            continue
        if active and len(row.values) >= 2 and not re.search(r"\btotal\b", label):
            out.append(row)
    return out


def _rows_after(rows: list[ParsedRow], anchor: ParsedRow | None) -> list[ParsedRow]:
    if anchor is None:
        return []
    for index, row in enumerate(rows):
        if row is anchor:
            return rows[index + 1:]
    return []


def check_income_statement(rows: list[ParsedRow]) -> list[Check]:
    total_income = _find(rows, r"\btotal income\b")
    revenue = _find(rows, r"\brevenue from operations\b")
    other_income = _find(rows, r"\bother income\b")
    total_expenses = _find(rows, r"\btotal expenses?\b")
    pbt = _find(rows, r"\bprofit before tax\b")
    tax = _find(rows, r"\btotal tax expense\b|\btax expense\b")
    net = _find(rows, r"^profit after tax$|^net profit$|\bprofit for the year\b")
    if net is None:
        net = _find(rows, r"\b(?:net )?profit for the (?:year|period)\b")

    expense_items = _section_items(rows, r"\btotal income\b",
                                   r"\bprofit before\b")
    exceptional = _find(rows, r"\bexceptional items net of tax\b", min_values=2)
    if exceptional is None:
        exceptional = _find(rows, r"\bexceptional items before tax\b", min_values=2)
    checks = [
        _sum_check("total_income_equals_parts", total_income, [revenue, other_income]),
        _sum_check("total_expenses_equals_items", total_expenses, expense_items),
    ]
    if total_income is not None and total_expenses is not None and pbt is not None:
        for column in _columns(total_income, total_expenses, pbt):
            expected = total_income.values[column] - total_expenses.values[column]
            if exceptional is not None and len(exceptional.values) > column:
                expected += exceptional.values[column]
            if not _approx(pbt.values[column], expected):
                checks.append(Check("profit_before_tax_equals_income_minus_expenses",
                                    "FAIL", f"column {column + 1}"))
                break
        else:
            checks.append(Check("profit_before_tax_equals_income_minus_expenses", "PASS"))
    else:
        checks.append(Check("profit_before_tax_equals_income_minus_expenses",
                            "NOT_CHECKABLE", "missing row"))

    after_pbt = _rows_after(rows, pbt)
    current_tax = _find(after_pbt, r"\bcurrent tax\b", min_values=2)
    deferred_tax = _find(after_pbt, r"\bdeferred tax\b", min_values=2)
    tax_parts = [row for row in (current_tax, deferred_tax) if row is not None]
    if not tax_parts:
        tax_parts = [row for row in (tax,) if row is not None and len(row.values) >= 2]
    if pbt is not None and net is not None and tax_parts:
        for column in _columns(pbt, net, *tax_parts):
            expected = pbt.values[column] - sum(
                (part.values[column] for part in tax_parts), Decimal("0"))
            if not _approx(net.values[column], expected):
                checks.append(Check("profit_after_tax_equals_parts",
                                    "FAIL", f"column {column + 1}"))
                break
        else:
            checks.append(Check("profit_after_tax_equals_parts", "PASS"))
    else:
        checks.append(Check("profit_after_tax_equals_parts",
                            "NOT_CHECKABLE", "missing row"))
    return checks


def _signed(row: ParsedRow | None, outflow: bool = False) -> ParsedRow | None:
    if row is None or not outflow:
        return row
    if any(value > 0 for value in row.values):
        return ParsedRow(row.label, tuple(-value if value > 0 else value
                                          for value in row.values))
    return row


def _is_outflow(label: str) -> bool:
    return bool(re.search(r"\bused in\b", label, re.I)
                and not re.search(r"\bgenerated\b", label, re.I))


def _flow_sum_check(net_change: ParsedRow | None,
                    flows: Sequence[ParsedRow | None]) -> Check:
    present = [flow for flow in flows if flow is not None and flow.values]
    if net_change is None or not net_change.values or not present:
        return Check("net_change_equals_operating_investing_financing",
                     "NOT_CHECKABLE", "missing row")
    variants: list[list[Decimal]] = []
    for column in _columns(net_change, *present):
        printed = [flow.values[column] for flow in present]
        variants.append(printed)
    attempts = [0, 1]
    for negate in attempts:
        matched = True
        for column, printed in enumerate(variants):
            expected = Decimal("0")
            for flow, value in zip(present, printed):
                value = value
                if negate and _is_outflow(flow.label) and value > 0:
                    value = -value
                expected += value
            if not _approx(net_change.values[column], expected):
                matched = False
                break
        if matched:
            return Check("net_change_equals_operating_investing_financing", "PASS")
    return Check("net_change_equals_operating_investing_financing", "FAIL")


def check_cash_flow(rows: list[ParsedRow]) -> list[Check]:
    operating = _find(rows, r"\bnet cash\b.*\boperating activities\b", min_values=2)
    investing = _find(rows, r"\bnet cash\b.*\binvesting activities\b", min_values=2)
    financing = _find(rows, r"\bnet cash\b.*\bfinancing activities\b", min_values=2)
    net_change = _find(rows, r"\bnet\b.*\b(?:decrease|increase)\b.*\bcash and cash equivalents\b",
                       min_values=2)
    opening = _find(rows, r"\bcash and cash equivalents at (the )?(beginning|commencement)\b"
                          r"|\bopening cash\b", min_values=2)
    closing = _find(rows, r"\bcash and cash equivalents at (the )?end\b|\bclosing cash\b",
                    min_values=2)
    exchange = _find(_rows_after(rows, net_change),
                     r"\beffect of exchange rate changes on cash\b", min_values=2)
    if exchange is None:
        exchange = _find(rows, r"\beffect of exchange rate changes on cash\b",
                         min_values=2)

    checks = [_flow_sum_check(net_change, [operating, investing, financing])]
    if opening is not None and closing is not None and net_change is not None:
        for column in _columns(opening, closing, net_change):
            expected = opening.values[column] + net_change.values[column]
            if exchange is not None and len(exchange.values) > column:
                expected += exchange.values[column]
            if not _approx(closing.values[column], expected):
                checks.append(Check("closing_cash_equals_opening_plus_change",
                                    "FAIL", f"column {column + 1}"))
                break
        else:
            checks.append(Check("closing_cash_equals_opening_plus_change", "PASS"))
    else:
        checks.append(Check("closing_cash_equals_opening_plus_change",
                            "NOT_CHECKABLE", "missing row"))
    return checks


CHECKERS = {
    "BS": check_balance_sheet,
    "PL": check_income_statement,
    "CF": check_cash_flow,
}


def evaluate_statement(company: str, checksum: str, basis: str, statement: str,
                       start: int, end: int, text: str) -> StatementResult:
    checks = CHECKERS[statement](parse_rows(text))
    return StatementResult(company, checksum, basis, statement, start, end,
                           _overall(checks), checks)


def load_span_text(extract_dir: Path, checksum: str, start: int, end: int) -> str:
    key = checksum[:-4] if checksum.endswith(".pdf") else checksum
    path = extract_dir / "page_extracts" / f"{key}.pages.jsonl"
    if not path.is_file():
        return ""
    parts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            if start <= int(payload["pdf_page_index"]) <= end:
                parts.append(payload["page_text"])
    return "\n".join(parts)


def run_map(statement_map: Path, extract_dir: Path) -> list[StatementResult]:
    results: list[StatementResult] = []
    with statement_map.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            checksum = (row.get("checksum") or row.get("document_checksum") or "").strip()
            text = load_span_text(extract_dir, checksum,
                                  int(row["start_pdf_page"]), int(row["end_pdf_page"]))
            results.append(evaluate_statement(
                row.get("company", ""), checksum, row["basis"], row["statement"],
                int(row["start_pdf_page"]), int(row["end_pdf_page"]), text))
    return results


def document_status(results: list[StatementResult]) -> dict[str, str]:
    out: dict[str, str] = {}
    for result in results:
        key = result.checksum
        current = out.get(key)
        if result.status == "FAIL" or current == "FAIL":
            out[key] = "FAIL"
        elif result.status == "PASS" or current == "PASS":
            out[key] = "PASS"
        else:
            out[key] = "NOT_CHECKABLE"
    return out


def write_results(path: Path, results: list[StatementResult]) -> None:
    fields = ["company", "checksum", "basis", "statement", "start_pdf_page",
              "end_pdf_page", "status", "checks"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({
                "company": result.company,
                "checksum": result.checksum,
                "basis": result.basis,
                "statement": result.statement,
                "start_pdf_page": result.start_pdf_page,
                "end_pdf_page": result.end_pdf_page,
                "status": result.status,
                "checks": "; ".join(f"{c.name}={c.status}" for c in result.checks),
            })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Arithmetic checks on proposed statement spans")
    parser.add_argument("--statement-map", default="data/qa/statement_map_proposed.csv")
    parser.add_argument("--storage-dir", default="data/corpus")
    parser.add_argument("--out", default="data/qa/arithmetic_results.csv")
    args = parser.parse_args(argv)
    results = run_map(Path(args.statement_map), Path(args.storage_dir) / "extract")
    write_results(Path(args.out), results)
    for checksum, status in document_status(results).items():
        company = next(r.company for r in results if r.checksum == checksum)
        print(f"{status:14} {company:12} {checksum[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
