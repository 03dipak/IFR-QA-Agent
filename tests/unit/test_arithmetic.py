"""Synthetic tests for statement-table arithmetic checks."""

from __future__ import annotations

from decimal import Decimal

from src.tables.arithmetic import (
    ParsedRow,
    check_balance_sheet,
    check_cash_flow,
    check_income_statement,
    document_status,
    evaluate_statement,
    parse_rows,
)


def _row(label: str, *values: str) -> ParsedRow:
    def _dec(value: str) -> Decimal:
        negative = value.startswith("(") and value.endswith(")")
        number = Decimal(value.strip("()"))
        return -number if negative else number

    return ParsedRow(label, tuple(_dec(v) for v in values))


def _status(checks) -> dict[str, str]:
    return {check.name: check.status for check in checks}


def test_balance_sheet_balances_and_subtotals_pass():
    rows = [
        _row("Total non-current assets", "100", "90"),
        _row("Total current assets", "50", "40"),
        _row("Total assets", "150", "130"),
        _row("Equity share capital", "20", "20"),
        _row("Other equity", "80", "70"),
        _row("Total equity", "100", "90"),
        _row("Total non-current liabilities", "30", "25"),
        _row("Total current liabilities", "20", "15"),
        _row("Total liabilities", "50", "40"),
        _row("Total equity and liabilities", "150", "130"),
    ]
    assert _status(check_balance_sheet(rows))["assets_equals_equity_and_liabilities"] == "PASS"
    assert _status(check_balance_sheet(rows))["total_equity_equals_parts"] == "PASS"


def test_balance_sheet_imbalance_fails():
    rows = [
        _row("Total assets", "150", "130"),
        _row("Total equity and liabilities", "149", "130"),
    ]
    checks = check_balance_sheet(rows)
    assert _status(checks)["assets_equals_equity_and_liabilities"] == "FAIL"


def test_balance_sheet_missing_rows_not_checkable():
    checks = check_balance_sheet([])
    assert all(check.status == "NOT_CHECKABLE" for check in checks)


def test_income_statement_subtotals_pass():
    rows = [
        _row("Revenue from operations", "100", "90"),
        _row("Other income", "10", "9"),
        _row("Total income", "110", "99"),
        _row("Expenses"),
        _row("Cost of materials consumed", "40", "35"),
        _row("Employee benefits expense", "20", "18"),
        _row("Total expenses", "60", "53"),
        _row("Profit before tax", "50", "46"),
        _row("Tax expense", "10", "9"),
        _row("Profit for the year", "40", "37"),
    ]
    checks = check_income_statement(rows)
    assert _status(checks)["total_income_equals_parts"] == "PASS"
    assert _status(checks)["total_expenses_equals_items"] == "PASS"
    assert _status(checks)["profit_after_tax_equals_parts"] == "PASS"


def test_income_statement_wrong_subtotal_fails():
    rows = [
        _row("Revenue from operations", "100", "90"),
        _row("Other income", "10", "9"),
        _row("Total income", "111", "99"),
    ]
    checks = check_income_statement(rows)
    assert _status(checks)["total_income_equals_parts"] == "FAIL"


def test_cash_flow_ties_opening_and_closing_cash():
    rows = [
        _row("Net cash from operating activities", "50", "40"),
        _row("Net cash used in investing activities", "(30)", "(20)"),
        _row("Net cash from financing activities", "5", "2"),
        _row("Net increase in cash and cash equivalents", "25", "22"),
        _row("Cash and cash equivalents at the beginning", "100", "78"),
        _row("Cash and cash equivalents at the end", "125", "100"),
    ]
    checks = check_cash_flow(rows)
    assert _status(checks)["net_change_equals_operating_investing_financing"] == "PASS"
    assert _status(checks)["closing_cash_equals_opening_plus_change"] == "PASS"


def test_cash_flow_broken_tie_fails():
    rows = [
        _row("Net cash from operating activities", "50", "40"),
        _row("Net cash from investing activities", "(30)", "(20)"),
        _row("Net cash from financing activities", "5", "2"),
        _row("Net increase in cash and cash equivalents", "30", "22"),
        _row("Cash and cash equivalents at the beginning", "100", "78"),
        _row("Cash and cash equivalents at the end", "130", "100"),
    ]
    checks = check_cash_flow(rows)
    assert _status(checks)["net_change_equals_operating_investing_financing"] == "FAIL"


def test_cash_flow_used_in_label_negates_positive_value():
    rows = [
        _row("Net cash from operating activities", "50", "40"),
        _row("Net cash used in investing activities", "30", "20"),
        _row("Net cash from financing activities", "5", "2"),
        _row("Net increase in cash and cash equivalents", "25", "22"),
        _row("Cash and cash equivalents at the beginning", "100", "78"),
        _row("Cash and cash equivalents at the end", "125", "100"),
    ]
    checks = check_cash_flow(rows)
    assert _status(checks)["net_change_equals_operating_investing_financing"] == "PASS"


def test_parse_rows_extracts_trailing_values_and_negative():
    text = "Revenue from operations 31 1,234.50 1,100.00\nOther income (20) (10)\n"
    rows = parse_rows(text)
    assert rows[0].values == (Decimal("1234.50"), Decimal("1100.00"))
    assert rows[1].values == (Decimal("-20"), Decimal("-10"))


def test_document_status_precedence():
    ok = evaluate_statement("c", "x", "consolidated", "BS", 1, 1,
                            "Total assets 1 1\nTotal equity and liabilities 1 1\n")
    bad = evaluate_statement("c", "x", "consolidated", "PL", 2, 2,
                             "Revenue from operations 1 1\nTotal income 2 2\n")
    empty = evaluate_statement("c", "y", "consolidated", "BS", 1, 1, "")
    assert document_status([ok]) == {"x": "PASS"}
    assert document_status([bad]) == {"x": "FAIL"}
    assert document_status([empty]) == {"y": "NOT_CHECKABLE"}
    assert document_status([ok, bad]) == {"x": "FAIL"}

def test_held_for_sale_and_equity_attributable():
    # Tata-style: "Assets held for sale" must be included (44.66)
    t = ("Total non current assets 202875.26\n"
         "Total current assets 70503.58\n"
         "Assets held for sale 44.66\n"
         "Total assets 273423.50\n"
         "Equity share capital 1247.44\n"
         "Other equity 90788.32\n"
         "Non controlling interests 396.98\n"
         "Total equity 92432.74\n"
         "Total equity attributable to equity holders 92035.76\n"
         "Total non current liabilities 82587.28\n"
         "Total current liabilities 98403.48\n"
         "Total liabilities 180990.76\n"
         "Total equity and liabilities 273423.50\n")
    checks = check_balance_sheet(parse_rows(t))
    d = {c.name: c.status for c in checks}
    assert d["assets_equal_current_and_non_current"] == "PASS", d
    # total equity must be the exact 92432.74 line, not the attributable line
    assert d["total_equity_equals_parts"] == "PASS", d

def test_pl_stops_at_profit_before_and_exact_pat():
    t = ("Revenue from operations 100 90\n"
         "Other income 5 4\n"
         "Total income 105 94\n"
         "Employee benefits expense 40 38\n"
         "Total expenses 40 38\n"
         "Profit before share of profit of associate and tax 65 56\n"
         "Profit before tax 65 56\n"
         "Total tax expense 10 9\n"
         "Profit for the year 55 47\n")
    checks = check_income_statement(parse_rows(t))
    d = {c.name: c.status for c in checks}
    assert d["total_expenses_equals_items"] == "PASS", d
    assert d["profit_before_tax_equals_income_minus_expenses"] == "PASS", d
    assert d["profit_after_tax_equals_parts"] == "PASS", d
