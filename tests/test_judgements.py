from euregsearch.qa.esma import QAEntry
from euregsearch.qa.judgements import (
    HELD_OUT_DIRECTIONS,
    HELD_OUT_INSTRUMENT,
    build_judgements,
    training_is_disjoint,
)

ENTRIES = [
    QAEntry(question="KID for professionals?", answer="a", language="en", source="s",
            citations=[("32014R1286", "2")]),
    QAEntry(question="Appropriateness?", answer="a", language="en", source="s",
            citations=[("32014L0065", "25")]),
    QAEntry(question="RTS scope?", answer="a", language="en", source="s",
            citations=[("32017R0653", "3")]),
]


def test_held_out_values_match_the_spec():
    assert HELD_OUT_INSTRUMENT == "32017R0653"
    assert HELD_OUT_DIRECTIONS == (("nl", "en"), ("de", "fr"))


def test_same_language_judgements_are_produced():
    same = [j for j in build_judgements(ENTRIES, ["en"]) if j.slice_name == "same_language"]
    assert len(same) == 2


def test_held_out_instrument_gets_its_own_slice():
    held = [j for j in build_judgements(ENTRIES, ["en"]) if j.slice_name == "held_out_instrument"]
    assert len(held) == 1
    assert held[0].relevant == [("32017R0653", "3")]


def test_cross_lingual_judgements_are_labelled_by_direction():
    judgements = build_judgements(ENTRIES, ["nl"])
    assert all(j.query_language == "en" and j.target_language == "nl" for j in judgements)
    assert {j.slice_name for j in judgements} <= {"cross_lingual", "held_out_instrument"}


def test_disjointness_passes_when_training_avoids_evaluation_articles():
    judgements = build_judgements(ENTRIES, ["en"])
    assert training_is_disjoint(judgements, [("32014R0600", "1")]) is True


def test_disjointness_fails_when_training_touches_an_evaluation_article():
    judgements = build_judgements(ENTRIES, ["en"])
    assert training_is_disjoint(judgements, [("32014L0065", "25")]) is False


def test_no_judgement_is_empty():
    assert all(j.relevant for j in build_judgements(ENTRIES, ["en", "nl"]))


def test_evaluation_articles_are_collected():
    from euregsearch.qa.judgements import evaluation_articles

    articles = evaluation_articles(build_judgements(ENTRIES, ["en"]))
    assert ("32014L0065", "25") in articles


def test_disjointness_now_guards_synthetic_questions_only():
    judgements = build_judgements(ENTRIES, ["en"])
    assert training_is_disjoint(judgements, [("32014R0600", "1")]) is True
    assert training_is_disjoint(judgements, [("32014L0065", "25")]) is False
