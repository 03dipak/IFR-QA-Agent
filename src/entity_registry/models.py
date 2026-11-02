"""Entity registry + fiscal calendar — data models.

Implements lld_entity_registry.txt: Company, CompanyNameHistory,
CompanyIdentifier, CompanyRelationship, AliasRecord, FiscalCalendar,
FiscalPeriod, EntityResolution, PeriodResolution, CompanyPeriodCoverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class LifecycleStatus(StrEnum):
    ACTIVE = "active"
    MERGED = "merged"
    ACQUIRED = "acquired"
    RENAMED = "renamed"
    DEMERGED = "demerged"
    DISSOLVED = "dissolved"


class RelationshipType(StrEnum):
    MERGED_INTO = "merged_into"
    ACQUIRED_BY = "acquired_by"
    RENAMED_TO = "renamed_to"
    DEMERGED_FROM = "demerged_from"
    SUCCESSOR_OF = "successor_of"


class AliasVerificationStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class ResolutionOutcome(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    GROUP_OR_PEER_SET = "group_or_peer_set"


class ConsolidatedAvailability(StrEnum):
    AVAILABLE = "available"
    NOT_AVAILABLE = "not_available"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class PeriodResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"


class PeriodType(StrEnum):
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    HALF_YEAR = "half_year"
    NINE_MONTH = "nine_month"
    TTM = "ttm"
    POINT_IN_TIME = "point_in_time"


class ResolutionIntent(StrEnum):
    LAST_FINANCIAL_YEAR = "last_financial_year"
    LATEST_ANNUAL_REPORT = "latest_annual_report"
    LATEST_QUARTER = "latest_quarter"
    LATEST_RESULTS = "latest_results"
    LAST_12_MONTHS = "last_12_months"
    THIS_FINANCIAL_YEAR = "this_financial_year"
    LATEST_AVAILABLE_FY = "latest_available_fy"


class CoverageStatus(StrEnum):
    PRESENT_USABLE = "present_usable"
    PRESENT_NOT_USABLE = "present_not_usable"
    PRESENT_NOT_EXTRACTED = "present_not_extracted"
    NOT_PRESENT = "not_present"
    EXCLUDED = "excluded"
    UNKNOWN = "unknown"


@dataclass
class EntityRegistryRow:
    """Backward-compatible alias for the entity registry row (LLD-13 title)."""
    company_id: str
    name: str = ""
    scrip_bse: str | None = None
    scrip_nse: str | None = None
    isin: str | None = None
    aliases: list[str] = field(default_factory=list)
    status: str = "active"
    fiscal_year_end: str = "03-31"
    corpus_presence: list[str] = field(default_factory=list)


@dataclass
class Company:
    company_id: str
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE
    effective_from: str = ""
    effective_to: str | None = None
    recorded_in_registry_version: str = "v0"
    recorded_at: str = ""


@dataclass
class CompanyNameHistory:
    company_id: str
    name: str
    effective_from: str = ""
    effective_to: str | None = None
    recorded_in_registry_version: str = "v0"


@dataclass
class CompanyIdentifier:
    company_id: str
    identifier_type: str  # scrip_bse | scrip_nse | isin
    value: str
    effective_from: str = ""
    effective_to: str | None = None
    recorded_in_registry_version: str = "v0"


@dataclass
class CompanyRelationship:
    relationship_id: str
    from_company_id: str
    to_company_id: str
    relationship_type: RelationshipType
    effective_from: str = ""
    recorded_in_registry_version: str = "v0"
    # an acquisition is NOT automatically a merger of identities


@dataclass
class AliasRecord:
    alias_id: str
    company_id: str
    alias_text: str
    normalized_text: str
    verification_status: AliasVerificationStatus = AliasVerificationStatus.UNVERIFIED
    effective_from: str = ""
    recorded_in_registry_version: str = "v0"


@dataclass
class FiscalCalendar:
    company_id: str
    fiscal_year_end_month: int  # 1-12
    fiscal_year_end_day: int
    effective_from: str = ""
    effective_to: str | None = None
    recorded_in_registry_version: str = "v0"


@dataclass
class FiscalPeriod:
    fiscal_period_id: str
    company_id: str
    period_type: PeriodType
    start_date: str
    end_date: str
    fiscal_year_label: str
    transition_status: str = "normal"  # normal | transition
    reporting_basis: str = "consolidated"
    resolution_status: PeriodResolutionStatus = PeriodResolutionStatus.RESOLVED
    reason: str = ""


@dataclass
class EntityResolution:
    matched_company_id: str | None = None
    canonical_company_id: str | None = None
    effective_date_context: str = ""
    outcome: ResolutionOutcome = ResolutionOutcome.NOT_FOUND
    candidate_company_ids: list[str] = field(default_factory=list)
    corpus_presence_status: str = ""
    chain: list[str] = field(default_factory=list)  # resolution chain notes


@dataclass
class PeriodResolution:
    period_type: PeriodType = PeriodType.ANNUAL
    fiscal_period_id: str | None = None
    gregorian_span: str = ""
    resolution_status: PeriodResolutionStatus = PeriodResolutionStatus.RESOLVED
    reason: str = ""
    intent: ResolutionIntent = ResolutionIntent.LAST_FINANCIAL_YEAR
    is_ttm_request: bool = False


@dataclass
class CompanyPeriodCoverage:
    freeze_id: str
    company_id: str
    fiscal_period_id: str | None = None
    document_type: str = ""
    eligible_document_count: int = 0
    accepted_document_count: int = 0
    usable_document_count: int = 0
    coverage_status: CoverageStatus = CoverageStatus.UNKNOWN
    evidence_document_ids: list[str] = field(default_factory=list)
    registry_version: str = "v0"
    ingestion_registry_version: str = "v0"


@dataclass
class RegistryVersion:
    registry_version_id: str
    created_at: str = ""
    snapshot_hash: str = ""
    change_record: list[dict[str, Any]] = field(default_factory=list)
    parent_version_id: str | None = None


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def normalize_company_name(name: str) -> str:
    """Deterministic normalisation: case/punctuation/legal suffixes stripped."""
    import re
    n = name.strip().lower()
    # remove common legal suffixes
    for suf in ["limited", "ltd", "ltd.", "private", "pvt", "inc", "corp", "corporation"]:
        n = n.replace(suf, "")
    n = re.sub(r"[^a-z0-9]", "", n)
    return n


def resolve_entity(name_or_id: str, registry: dict[str, Company],
                   aliases: dict[str, AliasRecord],
                   name_history: list[CompanyNameHistory]) -> EntityResolution:
    """Resolve a company from a name or id. Returns resolution with outcome."""
    # direct id lookup
    if name_or_id in registry:
        _ = registry[name_or_id]
        return EntityResolution(
            matched_company_id=name_or_id,
            canonical_company_id=name_or_id,
            effective_date_context="",
            outcome=ResolutionOutcome.RESOLVED,
            chain=[f"direct id {name_or_id}"],
        )
    # alias lookup (only VERIFIED aliases resolve)
    for a in aliases.values():
        norm = normalize_company_name(name_or_id)
        if (a.verification_status == AliasVerificationStatus.VERIFIED
                and a.normalized_text == norm):
            return EntityResolution(
                matched_company_id=a.company_id,
                canonical_company_id=a.company_id,
                effective_date_context="",
                outcome=ResolutionOutcome.RESOLVED,
                chain=[f"alias {a.alias_id}"],
            )
    # name history lookup
    norm = normalize_company_name(name_or_id)
    matches = [nh for nh in name_history if normalize_company_name(nh.name) == norm]
    if len(matches) == 1:
        return EntityResolution(
            matched_company_id=matches[0].company_id,
            canonical_company_id=matches[0].company_id,
            outcome=ResolutionOutcome.RESOLVED,
            chain=[f"name history {matches[0].company_id}"],
        )
    if len(matches) > 1:
        return EntityResolution(
            outcome=ResolutionOutcome.AMBIGUOUS,
            candidate_company_ids=[m.company_id for m in matches],
            chain=["multiple name-history matches"],
        )
    # group/peer set detection: if the query names a group/sector, return GROUP_OR_PEER_SET
    low = name_or_id.lower()
    if any(w in low for w in ["group", "sector", "peers", "industry"]):
        return EntityResolution(outcome=ResolutionOutcome.GROUP_OR_PEER_SET)
    return EntityResolution(outcome=ResolutionOutcome.NOT_FOUND)


def resolve_period(company_id: str, fiscal_calendar: FiscalCalendar,
                  intent: ResolutionIntent, corpus_latest_fy: str = "") -> PeriodResolution:
    """Resolve a period for a company based on intent and fiscal calendar."""
    # FY label names the year the fiscal year ENDS
    if intent == ResolutionIntent.LAST_FINANCIAL_YEAR:
        # latest completed annual; use corpus latest if provided
        fy = corpus_latest_fy or "FY25"
        start = f"{int(fy[2:])-1:02d}-04-01" if fy.startswith("FY") else ""
        end = f"{fy[2:]}-03-31"
        return PeriodResolution(
            period_type=PeriodType.ANNUAL,
            fiscal_period_id=f"{company_id}-{fy}-ANNUAL",
            gregorian_span=f"{start} to {end}",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            intent=intent,
        )
    if intent == ResolutionIntent.LATEST_QUARTER:
        return PeriodResolution(
            period_type=PeriodType.QUARTERLY,
            fiscal_period_id=f"{company_id}-LQ",
            gregorian_span="",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            intent=intent,
        )
    if intent == ResolutionIntent.LAST_12_MONTHS:
        return PeriodResolution(
            period_type=PeriodType.TTM,
            fiscal_period_id=f"{company_id}-TTM",
            gregorian_span="",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            intent=intent,
            is_ttm_request=True,
        )
    if intent == ResolutionIntent.THIS_FINANCIAL_YEAR:
        return PeriodResolution(
            period_type=PeriodType.ANNUAL,
            fiscal_period_id=f"{company_id}-CURRENT",
            gregorian_span="",
            resolution_status=PeriodResolutionStatus.RESOLVED,
            reason="current fiscal year, stated (not silently 'latest in corpus')",
            intent=intent,
        )
    return PeriodResolution(
        period_type=PeriodType.ANNUAL,
        resolution_status=PeriodResolutionStatus.RESOLVED,
        intent=intent,
    )


def is_out_of_corpus(resolution: EntityResolution) -> bool:
    return resolution.outcome == ResolutionOutcome.NOT_FOUND
