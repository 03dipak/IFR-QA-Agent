"""Stated-precision numeric comparison (lld_eval.txt §8, lld_normalisation §5).

Tolerance = half a unit of the last printed digit of the expected value,
scaled to absolute rupees by the expected value's unit suffix if any.
Unit conversion between expected and output is an exact power of ten; a
poor extraction never widens the band. Values render as printed.
"""

from __future__ import annotations

import re

from .answer_scoring import to_absolute_rupees

_MAG = [("crore", 1e7), ("lakh", 1e5), ("million", 1e6),
        ("billion", 1e9), ("thousand", 1e3)]


def decimals_of(text: str | None) -> int:
    if not text:
        return 0
    t = str(text).replace(",", "").lower()
    if "." not in t:
        return 0
    frac = t.split(".", 1)[1]
    frac = re.split(r"[^\d]", frac)[0]
    return len(frac) if frac else 0


def magnitude(text: str | None) -> float:
    t = (text or "").replace(",", "").lower()
    for needle, fac in _MAG:
        if needle in t:
            return fac
    return 1.0


def stated_precision_tolerance(expected_text: str | None) -> float:
    if not expected_text:
        return 0.0
    return 0.5 * (10.0 ** -decimals_of(expected_text)) * magnitude(expected_text)


def numeric_match(expected: str | None, output: str | None) -> bool:
    if not expected or not output:
        return False
    e = to_absolute_rupees(expected)
    o = to_absolute_rupees(output)
    if e is None or o is None:
        return False
    return abs(e - o) <= stated_precision_tolerance(expected)
