from euregsearch.qa.citations import parse_citations


def test_directive_with_article_and_paragraph():
    assert parse_citations("see Article 25(2) of Directive 2014/65/EU") == [("32014L0065", "25")]


def test_regulation_by_number_and_year():
    assert parse_citations("Article 8 of Regulation (EU) No 1286/2014") == [("32014R1286", "8")]


def test_delegated_regulation():
    assert parse_citations("Article 3 of Delegated Regulation (EU) 2017/653") == [("32017R0653", "3")]


def test_named_instrument_alias():
    assert parse_citations("Article 24 of MiFID II") == [("32014L0065", "24")]


def test_multiple_citations_in_one_sentence():
    text = "Article 25 of Directive 2014/65/EU and Article 8 of Regulation (EU) No 1286/2014"
    assert parse_citations(text) == [("32014L0065", "25"), ("32014R1286", "8")]


def test_duplicates_are_collapsed_preserving_order():
    text = "Article 25 of MiFID II, and again Article 25 of Directive 2014/65/EU"
    assert parse_citations(text) == [("32014L0065", "25")]


def test_article_with_letter_suffix():
    assert parse_citations("Article 4a of Directive 2014/65/EU") == [("32014L0065", "4a")]


def test_unknown_instrument_is_dropped_not_guessed():
    assert parse_citations("Article 5 of Directive 2009/65/EC") == []


def test_article_without_an_instrument_is_dropped():
    assert parse_citations("as set out in Article 12") == []


def test_empty_text():
    assert parse_citations("") == []
