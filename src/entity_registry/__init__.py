"""Entity registry + fiscal calendar.

Implements lld_entity_registry.txt — deterministic lookup for company and
period resolution. Owns EntityRegistryRow and PeriodResolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EntityRegistryRow:
    """One row in the entity registry.

    Resolution is a table lookup, deterministic. The table is versioned
    and frozen WITH the corpus.
    """
    company_id: str
    company_name: str
    scrip_bse: Optional[str] = None
    scrip_nse: Optional[str] = None
    isin: Optional[str] = None
    aliases: tuple[str, ...] = ()
    status: str = "active"
    fiscal_year_end: str = "03-31"
    corpus_presence: tuple[str, ...] = ()


@dataclass(frozen=True)
class PeriodResolution:
    """Resolved period for a company.

    gregorian_span is derived from fiscal_year_end.
    """
    period_label: str
    gregorian_span: str
    statement_type: str


def resolve_entity(
    company_id: str,
    registry: dict[str, EntityRegistryRow],
) -> EntityRegistryRow | None:
    """Resolve an entity by company_id. Returns None if not in corpus."""
    return registry.get(company_id)
