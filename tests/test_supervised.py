from euregsearch.provenance import ArticleRef
from euregsearch.qa.judgements import Judgement
from euregsearch.train.supervised import build_supervised_pairs, k_folds


def judgement(query, relevant, language="en"):
    return Judgement(query=query, query_language=language, target_language=language,
                     relevant=relevant, slice_name="same_language")


def ref(article, text, language="en"):
    return ArticleRef(celex="32014L0065", article=article, language=language,
                      version="consolidated", retrieved="2026-08-21", text=text)


def test_folds_partition_every_item_exactly_once():
    items = list(range(20))
    folds = k_folds(items, 5)
    assert len(folds) == 5
    seen = [x for _train, test in folds for x in test]
    assert sorted(seen) == items


def test_train_and_test_never_overlap():
    folds = k_folds(list(range(20)), 5)
    for train, test in folds:
        assert not set(train) & set(test)
        assert len(train) + len(test) == 20


def test_folds_are_deterministic_for_a_seed():
    assert k_folds(list(range(20)), 5, seed=1) == k_folds(list(range(20)), 5, seed=1)
    assert k_folds(list(range(20)), 5, seed=1) != k_folds(list(range(20)), 5, seed=2)


def test_uneven_split_still_covers_everything():
    folds = k_folds(list(range(13)), 5)
    seen = [x for _t, test in folds for x in test]
    assert sorted(seen) == list(range(13))


def test_pairs_link_a_question_to_the_text_of_its_cited_article():
    js = [judgement("when does suitability apply?", [("32014L0065", "25")])]
    refs = [ref("25", "suitability assessment rules"), ref("20", "otf requirements")]
    pairs = build_supervised_pairs(js, refs)
    assert pairs == [("when does suitability apply?", "suitability assessment rules", ("32014L0065", "25"))]


def test_every_chunk_of_a_cited_article_becomes_a_positive():
    js = [judgement("q", [("32014L0065", "4")])]
    refs = [ref("4", "first chunk"), ref("4", "second chunk")]
    assert len(build_supervised_pairs(js, refs)) == 2


def test_language_of_the_target_is_respected():
    js = [judgement("q", [("32014L0065", "25")], language="nl")]
    refs = [ref("25", "english text", "en"), ref("25", "nederlandse tekst", "nl")]
    pairs = build_supervised_pairs(js, refs)
    assert [p[1] for p in pairs] == ["nederlandse tekst"]


def test_a_citation_with_no_indexed_article_is_dropped():
    js = [judgement("q", [("99999X9999", "1")])]
    assert build_supervised_pairs(js, [ref("25", "text")]) == []
