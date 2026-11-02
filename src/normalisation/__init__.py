"""Indian numeric normalisation.

Implements lld_normalisation.txt — canonical owner of NormalisedValue and
the unit/rounding rules for Indian financial statements (crore/lakh/million,
Indian digit grouping, bracket negatives, zero-denominator rules).

This is the single source of truth for reading a financial number as it
appears in Indian statements. No consumer re-implements grouping or factor
arithmetic locally.

Exports NormalisedValue, UndefinedRatio, parse_value, format_indian_grouping,
divide, convert_unit.
"""

from .norm_core import (  # noqa: F401
    DEFAULT_EPSILON,
    SCALE_FACTORS,
    NormalisedValue,
    UndefinedRatio,
    parse_bracket_negative,
    parse_indian_grouping,
    parse_value,
)
from .norm_ops import (  # noqa: F401
    convert_unit,
    divide,
    format_indian_grouping,
)
