"""Unit tests for LLD-07..09 normalisation."""

from __future__ import annotations

from decimal import Decimal

from src.normalisation.norm_core import (
    Dimension,
    Exactness,
    NormalisedValue,
    ParseStatus,
    UndefinedRatio,
    classify_special_value,
    parse_bracket_negative,
    parse_indian_grouping,
    parse_value,
    resolve_alias,
    strict_numeric_grammar_ok,
)
from src.normalisation.norm_ops import divide, format_indian_grouping


def test_parse_indian_grouping():
    assert parse_indian_grouping("1,23,45,678") == "12345678"
    assert parse_indian_grouping("1,234.56") == "1234.56"


def test_parse_bracket_negative():
    assert parse_bracket_negative("(1,234)") == ("1,234", True)
    assert parse_bracket_negative("1,234") == ("1,234", False)


def test_resolve_alias():
    assert resolve_alias("crore") == "crore"
    assert resolve_alias("Cr.") == "crore"
    assert resolve_alias("lakhs") == "lakh"
    assert resolve_alias("mn") == "million"
    assert resolve_alias("'000") == "thousand"
    assert resolve_alias("lakh crore") == "lakh_crore"


def test_parse_value_crore():
    nv = parse_value("12.5", effective_unit="crore", dimension=Dimension.MONEY)
    assert nv.canonical_amount == Decimal("125000000")
    assert nv.scale_factor == Decimal("10000000")
    assert nv.parse_status == ParseStatus.PARSED


def test_parse_value_bracket_negative():
    nv = parse_value("(1,234)", effective_unit="plain")
    assert nv.canonical_amount == Decimal("-1234")
    assert nv.is_bracket_negative is True


def test_parse_value_lakh():
    nv = parse_value("5", effective_unit="lakh")
    assert nv.canonical_amount == Decimal("500000")


def test_parse_value_per_share_does_not_scale():
    nv = parse_value("12.5", effective_unit="crore", dimension=Dimension.PER_SHARE)
    assert nv.canonical_amount == Decimal("12.5")
    assert nv.scale_factor == Decimal("1")


def test_strict_grammar_malformed():
    assert strict_numeric_grammar_ok("1,2,3,4") is False
    assert strict_numeric_grammar_ok("12,,345") is False
    assert strict_numeric_grammar_ok(",123") is False
    assert strict_numeric_grammar_ok("123,") is False
    assert strict_numeric_grammar_ok("1.2.3") is False


def test_strict_grammar_valid():
    assert strict_numeric_grammar_ok("1,23,45,678") is True
    assert strict_numeric_grammar_ok("12,34,567.89") is True
    assert strict_numeric_grammar_ok("-1,234.56") is True


def test_classify_special_value():
    assert classify_special_value("-", convention_supports_nil=True) == "nil"
    assert classify_special_value("-", convention_supports_nil=False) == "unclear"
    assert classify_special_value("NIL") == "nil"
    assert classify_special_value("*") == "below_rounding"


def test_divide_normal():
    a = parse_value("10")
    b = parse_value("2")
    r = divide(a, b)
    assert isinstance(r, NormalisedValue)
    assert r.canonical_amount == Decimal("5")


def test_divide_zero():
    a = parse_value("10")
    b = parse_value("0")
    r = divide(a, b)
    assert isinstance(r, UndefinedRatio)


def test_format_indian_grouping():
    s = format_indian_grouping(Decimal("1234567"))
    assert "12" in s
    # just ensure no crash


def test_normalised_value_fields():
    nv = parse_value("100", source_ref="freeze:v1/table:t1")
    assert nv.source_ref == "freeze:v1/table:t1"
    assert nv.currency == "INR"
    assert nv.exactness in (Exactness.UNKNOWN_EXACTNESS,)


def test_parse_value_qa_flagged_on_bad():
    nv = parse_value("abc", effective_unit="crore")
    assert nv.parse_status == ParseStatus.QA_FLAGGED
