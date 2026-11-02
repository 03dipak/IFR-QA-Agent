"""Core normalisation types and functions.

Implements lld_normalisation.txt sections 3-4: NormalisedValue (expanded),
UnitAssertion, UnitResolution, UnitResolutionTrace, enums, parsing rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

# ---------------------------------------------------------------------------
# Closed enumerations (lld_normalisation.txt section 3)
# ---------------------------------------------------------------------------


class Dimension(StrEnum):
    MONEY = "money"
    PER_SHARE = "per_share"
    PERCENT = "percent"
    COUNT = "count"
    RATIO = "ratio"


class Exactness(StrEnum):
    EXACT_AS_PRINTED = "exact_as_printed"
    DISPLAYED_ROUNDED = "displayed_rounded"
    APPROXIMATE_SOURCE = "approximate_source"
    BELOW_REPORTED_PRECISION = "below_reported_precision"
    UNKNOWN_EXACTNESS = "unknown_exactness"


class ParseStatus(StrEnum):
    PARSED = "parsed"
    QA_FLAGGED = "qa_flagged"


class DimensionResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    QA_REQUIRED = "qa_required"


class CurrencyResolutionStatus(StrEnum):
    STATED = "stated"
    INHERITED = "inherited"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    NOT_STATED = "not_stated"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class SpecialValueStatus(StrEnum):
    VALUE = "value"
    NIL = "nil"
    NOT_APPLICABLE = "not_applicable"
    BELOW_ROUNDING = "below_rounding"
    MISSING = "missing"
    UNCLEAR = "unclear"


# ---------------------------------------------------------------------------
# Scale factors (closed alias table)
# ---------------------------------------------------------------------------

SCALE_FACTORS = {
    "crore": Decimal("10000000"),
    "lakh": Decimal("100000"),
    "million": Decimal("1000000"),
    "thousand": Decimal("1000"),
    "billion": Decimal("1000000000"),
    "lakh_crore": Decimal("1000000000000"),
    "plain": Decimal("1"),
    "units": Decimal("1"),
}

ALIAS_MAP = {
    "crore": "crore", "crores": "crore", "cr": "crore", "crs": "crore", "cr.": "crore",
    "lakh": "lakh", "lakhs": "lakh", "lacs": "lakh",
    "million": "million", "mn": "million",
    "thousand": "thousand", "000": "thousand", "'000": "thousand",
    "billion": "billion", "bn": "billion",
    "lakh crore": "lakh_crore",
    "plain": "plain", "units": "units",
}

DEFAULT_EPSILON = Decimal("0.001")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SourcePrecision:
    decimal_places: int = 0
    rounding_mode: str = "half_up"
    rounding_quantum: str = "0.01"


@dataclass
class UnitAssertion:
    assertion_id: str
    source_type: str  # table_header | footnote | row_signal | cell_signal | document
    source_ref: str
    scope_type: str  # DOCUMENT | TABLE | COLUMN | ROW | CELL
    scope_target: str
    printed_text: str
    parsed_unit: str
    parsed_dimension: str
    parsed_currency: str
    confidence: float = 0.0
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED


@dataclass
class UnitResolutionTrace:
    assertion_id: str
    scope_type: str
    source_type: str
    source_ref: str
    source_text: str
    parsed_unit: str
    parsed_dimension: str
    parsed_currency: str
    scale_factor: Decimal
    verification_status: VerificationStatus
    applied: bool
    rejection_reason: str = ""
    normalization_rules_version: str = "v0"


@dataclass
class UnitResolution:
    printed_unit_text: str
    parsed_scale_factor: Decimal
    resolved_dimension: Dimension
    resolved_currency: str
    dimension_resolution_status: DimensionResolutionStatus
    resolution_status: ResolutionStatus
    applied_assertion_id: str | None = None
    source_ref: str = ""


@dataclass
class NormalisedValue:
    raw: str
    dimension: Dimension = Dimension.MONEY
    dimension_resolution_status: DimensionResolutionStatus = DimensionResolutionStatus.RESOLVED
    currency: str = "INR"
    measurement_unit: str = "INR"
    effective_unit: str = "plain"
    scale_factor: Decimal = Decimal("1")
    canonical_amount: Decimal = Decimal("0")
    is_bracket_negative: bool = False
    override_chain: list[UnitResolutionTrace] = field(default_factory=list)
    source_precision: SourcePrecision = field(default_factory=SourcePrecision)
    exactness: Exactness = Exactness.UNKNOWN_EXACTNESS
    parse_status: ParseStatus = ParseStatus.PARSED
    source_ref: str = ""
    currency_resolution_status: CurrencyResolutionStatus = CurrencyResolutionStatus.STATED
    special_value_status: SpecialValueStatus = SpecialValueStatus.VALUE


@dataclass(frozen=True)
class UndefinedRatio:
    reason: str


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def resolve_alias(text: str) -> str | None:
    key = text.strip().lower().rstrip(".")
    # handle "lakh crore"
    if "lakh crore" in key:
        return "lakh_crore"
    return ALIAS_MAP.get(key)


def parse_indian_grouping(text: str) -> str:
    """Strip grouping separators (commas, spaces) losslessly."""
    cleaned = re.sub(r"[,\s\u00a0\u202f]", "", text)
    return cleaned


def parse_bracket_negative(text: str) -> tuple[str, bool]:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1], True
    return text, False


def _unicode_normalise_minus(text: str) -> str:
    return text.replace("\u2212", "-").replace("\u2013", "-")


def strict_numeric_grammar_ok(text: str) -> bool:
    """Validate strict numeric grammar after stripping grouping."""
    t = _unicode_normalise_minus(text.strip())
    # remove grouping commas
    # check for scientific notation
    if "e" in t.lower():
        return False
    # strip leading sign
    if t and t[0] in "+-":
        t = t[1:]
    # remove commas
    t2 = t.replace(",", "")
    if t2.count(".") > 1:
        return False
    # validate grouping (presentation): all groups except first same width
    # (2 for Indian or 3 for Western), first group <= that width
    if "," in t:
        intpart_for_groups = t.split(".")[0]
        groups = intpart_for_groups.split(",")
        if len(groups) > 1:
            # Western: all groups 3 digits, first <=3
            western = all(len(g) == 3 for g in groups) and len(groups[0]) <= 3
            # Indian: first <=2, middle groups 2, last group 3
            indian = (len(groups[-1]) == 3 and all(len(g) == 2 for g in groups[1:-1])
                      and 1 <= len(groups[0]) <= 2)
            if not (western or indian):
                return False
            for g in groups:
                if not g.isdigit():
                    return False
    # reject spaces
    if " " in t2:
        return False
    # reject currency symbols
    if any(c in t2 for c in "$€£¥₹"):
        return False
    # validate chars
    allowed = set("0123456789.,")
    if not all(c in allowed for c in t2):
        return False
    # reject malformed grouping like empty parts
    if t2.startswith(",") or t2.endswith(",") or ",," in t2:
        return False
    if "." in t2:
        intpart, frac = t2.split(".")
        if frac == "" :
            return False
    else:
        intpart = t2
    if intpart == "" and "." not in t2:
        return False
    return True


def parse_value(
    raw: str,
    effective_unit: str = "plain",
    is_bracket_negative: bool = False,
    dimension: Dimension = Dimension.MONEY,
    currency: str = "INR",
    source_precision: SourcePrecision | None = None,
    exactness: Exactness = Exactness.UNKNOWN_EXACTNESS,
    source_ref: str = "",
    normalization_rules_version: str = "v0",
) -> NormalisedValue:
    """Parse a raw printed figure into a NormalisedValue."""
    text = _unicode_normalise_minus(raw)
    cleaned, bracket_neg = parse_bracket_negative(text)
    if not strict_numeric_grammar_ok(cleaned):
        # qa-flagged
        return NormalisedValue(
            raw=raw,
            dimension=dimension,
            currency=currency,
            measurement_unit=currency if dimension == Dimension.MONEY else dimension.value,
            effective_unit=effective_unit,
            scale_factor=Decimal("1"),
            canonical_amount=Decimal("0"),
            is_bracket_negative=is_bracket_negative or bracket_neg,
            source_precision=source_precision or SourcePrecision(),
            exactness=Exactness.UNKNOWN_EXACTNESS,
            parse_status=ParseStatus.QA_FLAGGED,
            source_ref=source_ref,
        )
    digits = parse_indian_grouping(cleaned)
    try:
        magnitude = Decimal(digits)
    except Exception:
        return NormalisedValue(
            raw=raw,
            dimension=dimension,
            currency=currency,
            measurement_unit=currency if dimension == Dimension.MONEY else dimension.value,
            effective_unit=effective_unit,
            scale_factor=Decimal("1"),
            canonical_amount=Decimal("0"),
            is_bracket_negative=is_bracket_negative or bracket_neg,
            source_precision=source_precision or SourcePrecision(),
            exactness=Exactness.UNKNOWN_EXACTNESS,
            parse_status=ParseStatus.QA_FLAGGED,
            source_ref=source_ref,
        )

    if is_bracket_negative or bracket_neg:
        magnitude = -magnitude

    # scale policy per dimension
    scales = False
    if dimension == Dimension.MONEY:
        scales = True
    elif dimension == Dimension.COUNT:
        # count scales only when explicitly declared (effective_unit != plain)
        scales = effective_unit != "plain"
    # per_share, percent, ratio never scale

    unit_key = resolve_alias(effective_unit) or effective_unit
    if unit_key in SCALE_FACTORS:
        factor = SCALE_FACTORS[unit_key]
    else:
        factor = Decimal("1")
    if unit_key not in SCALE_FACTORS:
        factor = Decimal("1")
    if not scales:
        factor = Decimal("1")

    canonical = magnitude * factor

    return NormalisedValue(
        raw=raw,
        dimension=dimension,
        currency=currency,
        measurement_unit=(currency if dimension == Dimension.MONEY
                         else ("INR/share" if dimension == Dimension.PER_SHARE
                               else dimension.value)),
        effective_unit=effective_unit,
        scale_factor=factor,
        canonical_amount=canonical,
        is_bracket_negative=is_bracket_negative or bracket_neg,
        source_precision=source_precision or SourcePrecision(),
        exactness=exactness,
        parse_status=ParseStatus.PARSED,
        source_ref=source_ref,
    )


def classify_special_value(text: str, convention_supports_nil: bool = False) -> SpecialValueStatus:
    t = text.strip()
    if t in ("", "-", "–", "—"):
        if convention_supports_nil:
            return SpecialValueStatus.NIL
        return SpecialValueStatus.UNCLEAR
    if t.upper() in ("NIL", "N/A", "NA"):
        if t.upper() == "NIL":
            return SpecialValueStatus.NIL
        if t.upper() in ("N/A", "NA"):
            return SpecialValueStatus.NOT_APPLICABLE
    if t in ("*", "**"):
        return SpecialValueStatus.BELOW_ROUNDING
    return SpecialValueStatus.VALUE


def format_indian_grouping_decimal(value: Decimal) -> str:
    """Render a Decimal in Indian grouping (re-export from norm_ops)."""
    from .norm_ops import format_indian_grouping
    return format_indian_grouping(value)
