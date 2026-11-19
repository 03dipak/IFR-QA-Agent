from src.resolver.query_resolver import (
    load_company_registry,
    resolve_basis,
    resolve_companies,
    resolve_metric,
    resolve_period,
    resolve_query,
)


def test_company_alias_resolution():
    reg = load_company_registry()
    assert resolve_companies("Between Deepak Nitrite and Srf on total assets", reg) == [
        "deepak_nitrite", "srf"]


def test_company_ticker_resolution():
    reg = load_company_registry()
    assert resolve_companies("What was TCS revenue?", reg) == ["tcs"]


def test_period_fy_and_quarter():
    slots = resolve_period("For FY2026 and Q3 FY25")
    toks = {s.token for s in slots}
    assert "FY2026" in toks
    assert "Q3FY2025" in toks


def test_period_half_and_nine_months():
    slots = resolve_period("H1 FY25 and nine months ended 31 December 2024")
    toks = {s.token for s in slots}
    assert "H1FY2025" in toks
    assert "9MFY2025" in toks


def test_basis_explicit():
    assert resolve_basis("consolidated results") == ("consolidated", "consolidated", "explicit")
    assert resolve_basis("standalone results") == ("standalone", "standalone", "explicit")
    assert resolve_basis("total assets") == (None, None, "unspecified")


def test_metric_vocabulary():
    assert resolve_metric("net profit") == "net_profit"
    assert resolve_metric("PAT") == "net_profit"
    assert resolve_metric("profit for the period") == "net_profit"
    assert resolve_metric("total assets") == "total_assets"


def test_resolve_query():
    q = "For FY2026 consolidated results, what was Deepak Nitrite net profit?"
    r = resolve_query(q)
    assert r.company_ids == ["deepak_nitrite"]
    assert r.periods[0].token == "FY2026"
    assert r.basis == "consolidated"
    assert r.metric_id == "net_profit"
