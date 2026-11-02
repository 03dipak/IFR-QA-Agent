"""Normalisation operations.

Implements lld_normalisation.txt sections 4-5:
- format_indian_grouping: render Decimal in Indian grouping
- divide: ratio with UNDEFINED_RATIO for zero/near-zero denominators
- convert_unit: explicit unit conversion between printing families
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from .norm_core import (
    DEFAULT_EPSILON,
    SCALE_FACTORS,
    NormalisedValue,
    UndefinedRatio,
)


def format_indian_grouping(value: Decimal) -> str:
    """Render a Decimal value in Indian digit grouping (1,23,45,678).

    Values display at 2 decimals; magnitudes >= 1e6 display as integers.
    Exactness flag is kept on the record via canonical_amount.
    """
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    if abs(rounded) >= Decimal("1000000"):
        rounded = rounded.quantize(Decimal("1"), rounding=ROUND_HALF_UP)

    sign = "-" if rounded < 0 else ""
    abs_val = abs(rounded)

    int_part = int(abs_val)
    dec_part = abs_val - int_part

    int_str = str(int_part)
    if len(int_str) <= 3:
        grouped = int_str
    else:
        last_three = int_str[-3:]
        rest = int_str[:-3]
        groups: list[str] = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        groups.insert(0, rest)
        groups.append(last_three)
        grouped = ",".join(groups)

    result = f"{sign}{grouped}"
    if dec_part > 0:
        dec_str = str(dec_part)[2:]
        result += f".{dec_str}"

    return result


def divide(
    numerator: NormalisedValue,
    denominator: NormalisedValue,
    epsilon: Decimal = DEFAULT_EPSILON,
) -> NormalisedValue | UndefinedRatio:
    """Divide two NormalisedValues.

    Returns UndefinedRatio when denominator is zero or near-zero
    (below epsilon). Never crashes, never fabricates 0.
    """
    denom_amount = abs(denominator.canonical_amount)
    if denom_amount < epsilon:
        return UndefinedRatio(
            reason=(
                f"Denominator {denominator.raw} is zero or near-zero "
                f"(< {epsilon})"
            )
        )

    ratio = numerator.canonical_amount / denominator.canonical_amount
    return NormalisedValue(
        raw=f"({numerator.raw} / {denominator.raw})",
        effective_unit="plain",
        canonical_amount=ratio,
        is_bracket_negative=False,
    )


def convert_unit(
    value: NormalisedValue,
    target_unit: str,
) -> NormalisedValue:
    """Convert a NormalisedValue from one printing family to another.

    No implicit currency conversion. Two values compare/divide/sum only
    when their printing-family units match or are converted once explicitly.

    Args:
        value: The NormalisedValue to convert.
        target_unit: Target printing family (crore, lakh, million, plain).

    Returns:
        New NormalisedValue with canonical_amount in target unit.
    """
    source_scale = SCALE_FACTORS.get(value.effective_unit, Decimal("1"))
    target_scale = SCALE_FACTORS.get(target_unit, Decimal("1"))

    converted_amount = value.canonical_amount / source_scale * target_scale

    return NormalisedValue(
        raw=value.raw,
        effective_unit=target_unit,
        canonical_amount=converted_amount,
        is_bracket_negative=value.is_bracket_negative,
    )
