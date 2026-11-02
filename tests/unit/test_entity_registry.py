"""Unit tests for LLD-13 entity registry."""

from __future__ import annotations

from src.entity_registry import (
    AliasRecord,
    AliasVerificationStatus,
    Company,
    CompanyRelationship,
    EntityRegistryRow,
    FiscalCalendar,
    PeriodType,
    RelationshipType,
    ResolutionIntent,
    ResolutionOutcome,
    is_out_of_corpus,
    normalize_company_name,
    resolve_entity,
    resolve_period,
)


def test_normalize_company_name():
    assert normalize_company_name("Reliance Industries Ltd.") == "relianceindustries"
    assert normalize_company_name("TCS") == "tcs"


def test_resolve_entity_direct():
    reg = {"C1": Company(company_id="C1")}
    r = resolve_entity("C1", reg, {}, [])
    assert r.outcome == ResolutionOutcome.RESOLVED
    assert r.matched_company_id == "C1"


def test_resolve_entity_alias_verified():
    reg = {"C1": Company(company_id="C1")}
    vs = AliasVerificationStatus.VERIFIED
    alias = AliasRecord(alias_id="a1", company_id="C1", alias_text="Reliance",
                        normalized_text="reliance", verification_status=vs)
    r = resolve_entity("Reliance", reg, {"a1": alias}, [])
    assert r.outcome == ResolutionOutcome.RESOLVED
    assert r.matched_company_id == "C1"


def test_resolve_entity_alias_unverified_not_resolve():
    reg = {"C1": Company(company_id="C1")}
    vs = AliasVerificationStatus.UNVERIFIED
    alias = AliasRecord(alias_id="a1", company_id="C1", alias_text="Reliance",
                        normalized_text="reliance", verification_status=vs)
    r = resolve_entity("Reliance", reg, {"a1": alias}, [])
    assert r.outcome != ResolutionOutcome.RESOLVED


def test_resolve_entity_ambiguous():
    _ = type("NH", (), {"company_id": "C1", "name": "Foo Ltd"})()
    _ = type("NH", (), {"company_id": "C2", "name": "Foo Ltd"})()
    reg = {}
    from src.entity_registry.models import CompanyNameHistory
    h1 = CompanyNameHistory(company_id="C1", name="Foo Ltd")
    h2 = CompanyNameHistory(company_id="C2", name="Foo Ltd")
    r = resolve_entity("Foo Ltd", reg, {}, [h1, h2])
    assert r.outcome == ResolutionOutcome.AMBIGUOUS
    assert set(r.candidate_company_ids) == {"C1", "C2"}


def test_resolve_entity_not_found():
    r = resolve_entity("Nobody", {}, {}, [])
    assert r.outcome == ResolutionOutcome.NOT_FOUND
    assert is_out_of_corpus(r) is True


def test_resolve_entity_group():
    r = resolve_entity("banking sector peers", {}, {}, [])
    assert r.outcome == ResolutionOutcome.GROUP_OR_PEER_SET


def test_resolve_period_last_fy():
    from src.entity_registry.models import FiscalCalendar
    cal = FiscalCalendar(company_id="C1", fiscal_year_end_month=3, fiscal_year_end_day=31)
    pr = resolve_period("C1", cal, ResolutionIntent.LAST_FINANCIAL_YEAR, corpus_latest_fy="FY25")
    assert pr.period_type == PeriodType.ANNUAL
    assert pr.is_ttm_request is False
    assert "FY25" in pr.fiscal_period_id


def test_resolve_period_ttm():
    cal = FiscalCalendar(company_id="C1", fiscal_year_end_month=3, fiscal_year_end_day=31)
    pr = resolve_period("C1", cal, ResolutionIntent.LAST_12_MONTHS)
    assert pr.period_type == PeriodType.TTM
    assert pr.is_ttm_request is True


def test_resolve_period_this_fy_stated():
    cal = FiscalCalendar(company_id="C1", fiscal_year_end_month=3, fiscal_year_end_day=31)
    pr = resolve_period("C1", cal, ResolutionIntent.THIS_FINANCIAL_YEAR)
    assert "current fiscal year" in pr.reason or pr.reason != ""


def test_entity_registry_row_fields():
    row = EntityRegistryRow(company_id="C1", name="Acme", fiscal_year_end="03-31")
    assert row.company_id == "C1"
    assert row.fiscal_year_end == "03-31"


def test_relationship_types():
    rel = CompanyRelationship(relationship_id="r1", from_company_id="A",
                              to_company_id="B", relationship_type=RelationshipType.MERGED_INTO)
    assert rel.relationship_type == RelationshipType.MERGED_INTO


def test_coverage_status():
    from src.entity_registry import CompanyPeriodCoverage, CoverageStatus
    cov = CompanyPeriodCoverage(freeze_id="f1", company_id="C1",
                                coverage_status=CoverageStatus.PRESENT_USABLE)
    assert cov.coverage_status == CoverageStatus.PRESENT_USABLE
