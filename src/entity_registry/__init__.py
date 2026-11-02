"""Entity registry + fiscal calendar.

Implements lld_entity_registry.txt — deterministic lookup for company and
period resolution. Owns EntityRegistryRow and PeriodResolution.
"""

from .models import (
    AliasRecord,
    AliasVerificationStatus,
    Company,
    CompanyIdentifier,
    CompanyNameHistory,
    CompanyPeriodCoverage,
    CompanyRelationship,
    ConsolidatedAvailability,
    CoverageStatus,
    EntityRegistryRow,
    EntityResolution,
    FiscalCalendar,
    FiscalPeriod,
    LifecycleStatus,
    PeriodResolution,
    PeriodResolutionStatus,
    PeriodType,
    RegistryVersion,
    RelationshipType,
    ResolutionIntent,
    ResolutionOutcome,
    is_out_of_corpus,
    normalize_company_name,
    resolve_entity,
    resolve_period,
)
from .periods import (
    common_contract_for_headers,
    compute_gregorian_span,
    fiscal_year_label,
    periods_comparable,
    resolve_period_with_calendar,
)

__all__ = [
    "AliasRecord", "AliasVerificationStatus", "Company", "CompanyIdentifier",
    "EntityRegistryRow",
    "CompanyNameHistory", "CompanyPeriodCoverage", "CompanyRelationship",
    "ConsolidatedAvailability", "CoverageStatus", "EntityResolution",
    "FiscalCalendar", "FiscalPeriod", "LifecycleStatus", "PeriodResolution",
    "PeriodResolutionStatus", "PeriodType", "RegistryVersion", "RelationshipType",
    "ResolutionIntent", "ResolutionOutcome", "common_contract_for_headers",
    "compute_gregorian_span", "fiscal_year_label", "is_out_of_corpus",
    "normalize_company_name", "periods_comparable", "resolve_entity",
    "resolve_period", "resolve_period_with_calendar",
]
