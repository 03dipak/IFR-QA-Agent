from src.agent.tools import _normalize_value_to_crore, compute_margin


def test_lakh_to_crore_normalization():
    value, unit, normalized = _normalize_value_to_crore("150", "Rs. in lakh")
    assert normalized is True
    assert unit == "Rs. crore"
    assert value == "1.5"


def test_percent_formatting_has_no_currency():
    result = compute_margin(25, 200)
    assert result["margin_percent"] == 12.5
    # formatting contract: margins/growth are represented as numeric percent values
    assert isinstance(result["margin_percent"], float)
