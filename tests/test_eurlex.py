from pathlib import Path

from euregsearch.corpus.eurlex import segment_articles

FIXTURE = Path(__file__).parent / "fixtures" / "eurlex_sample.html"
ARGS = dict(celex="32014L0065", language="en", version="consolidated", retrieved="2026-08-21")


def articles():
    return segment_articles(FIXTURE.read_text(encoding="utf-8"), **ARGS)


def test_finds_every_article():
    assert [a.article for a in articles()] == ["1", "2", "25"]


def test_captures_the_anchor_for_deep_linking():
    assert articles()[0].anchor == "d1e1592-349-1"


def test_body_text_is_attached_to_its_article():
    assert "scope of this Directive" in articles()[0].text


def test_body_stops_at_the_next_article():
    assert "Definitions" not in articles()[0].text


def test_provenance_is_carried_onto_every_article():
    for ref in articles():
        assert ref.celex == "32014L0065"
        assert ref.version == "consolidated"
        assert ref.is_complete()


def test_empty_html_yields_no_articles():
    assert segment_articles("<html></html>", **ARGS) == []
