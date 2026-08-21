from euregsearch.provenance import ArticleRef
from euregsearch.train.pairs import build_cross_lingual_pairs, build_synthetic_pairs


def ref(celex, article, language, text):
    return ArticleRef(celex=celex, article=article, language=language, version="consolidated",
                      retrieved="2026-08-21", text=text)


REFS = [
    ref("32014L0065", "25", "en", "Assessment of suitability and appropriateness"),
    ref("32014L0065", "25", "nl", "Beoordeling van geschiktheid en passendheid"),
    ref("32014L0065", "25", "de", "Beurteilung der Eignung und Angemessenheit"),
    ref("32017R0653", "3", "en", "Held out instrument article"),
    ref("32017R0653", "3", "nl", "Uitgesloten instrument artikel"),
]


def test_cross_lingual_pairs_align_the_same_article():
    pairs = build_cross_lingual_pairs(REFS)
    assert all(key == ("32014L0065", "25") for _a, _p, key in pairs)


def test_held_out_instrument_produces_no_training_pairs():
    keys = {key for _a, _p, key in build_cross_lingual_pairs(REFS)}
    assert ("32017R0653", "3") not in keys


def test_held_out_directions_are_excluded():
    pairs = build_cross_lingual_pairs(REFS)
    texts = {(a, p) for a, p, _k in pairs}
    nl = "Beoordeling van geschiktheid en passendheid"
    en = "Assessment of suitability and appropriateness"
    assert (nl, en) not in texts


def test_a_permitted_direction_is_present():
    pairs = build_cross_lingual_pairs(REFS)
    texts = {(a, p) for a, p, _k in pairs}
    en = "Assessment of suitability and appropriateness"
    nl = "Beoordeling van geschiktheid en passendheid"
    assert (en, nl) in texts


def test_synthetic_pairs_reference_the_article():
    pairs = build_synthetic_pairs(REFS)
    assert all(key[0] != "32017R0653" for _a, _p, key in pairs)
    assert any("Article 25" in anchor for anchor, _p, _k in pairs)


def test_no_pairs_from_an_empty_corpus():
    assert build_cross_lingual_pairs([]) == []
    assert build_synthetic_pairs([]) == []


def test_synthetic_pairs_skip_excluded_evaluation_articles():
    pairs = build_synthetic_pairs(REFS, exclude={("32014L0065", "25")})
    assert all(key != ("32014L0065", "25") for _a, _p, key in pairs)


def test_synthetic_pairs_without_exclusions_are_unchanged():
    assert len(build_synthetic_pairs(REFS)) == len(build_synthetic_pairs(REFS, exclude=set()))
