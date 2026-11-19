"""Deterministic query resolver (LLD-13/LLD-retrieval section 6). No LLM."""

from .query_resolver import (
    CompanyRegistry,
    PeriodSlot,
    ResolutionResult,
    load_company_registry,
    resolve_basis,
    resolve_companies,
    resolve_metric,
    resolve_period,
    resolve_query,
)

__all__ = [
    "CompanyRegistry", "PeriodSlot", "ResolutionResult", "load_company_registry",
    "resolve_basis", "resolve_companies", "resolve_metric", "resolve_period",
    "resolve_query",
]
