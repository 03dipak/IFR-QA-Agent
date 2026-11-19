from src.rag_ingestion.tokenizer import tokenize, TOKENIZER_VERSION


def test_indian_grouping():
    assert tokenize("1,23,456.78") == ["123456.78"]


def test_bracket_negative():
    assert tokenize("(1,234)") == ["-1234"]


def test_lakh_crore_words():
    toks = tokenize("5 lakh 1.25 crore")
    assert "lakh" in toks
    assert "crore" in toks


def test_one_point_two_five_not_one_twenty_five():
    assert tokenize("1.25") == ["1.25"]
    assert tokenize("125") == ["125"]
    assert "125" not in tokenize("1.25")
