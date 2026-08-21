import pytest
from pydantic import ValidationError

from euregsearch.provenance import EURLEX_ATTRIBUTION, ArticleRef

REF = dict(celex="32014L0065", article="25", language="en",
           version="consolidated", retrieved="2026-08-21",
           anchor="d1e1592-349-1", text="Article 25 text")


def test_deep_link_includes_celex_language_and_anchor():
    link = ArticleRef(**REF).deep_link()
    assert "CELEX:32014L0065" in link
    assert "/EN/" in link
    assert link.endswith("#d1e1592-349-1")


def test_deep_link_without_anchor_still_resolves_to_the_document():
    link = ArticleRef(**{**REF, "anchor": None}).deep_link()
    assert "CELEX:32014L0065" in link
    assert "#" not in link


def test_complete_reference_is_complete():
    assert ArticleRef(**REF).is_complete()


def test_reference_without_anchor_is_still_complete():
    assert ArticleRef(**{**REF, "anchor": None}).is_complete()


def test_blank_article_is_rejected():
    with pytest.raises(ValidationError):
        ArticleRef(**{**REF, "article": "  "})


def test_language_must_be_one_of_the_four():
    with pytest.raises(ValidationError):
        ArticleRef(**{**REF, "language": "es"})


def test_version_must_be_original_or_consolidated():
    with pytest.raises(ValidationError):
        ArticleRef(**{**REF, "version": "draft"})


def test_attribution_names_the_reuse_decision():
    assert "2011/833/EU" in EURLEX_ATTRIBUTION
