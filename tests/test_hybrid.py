from euregsearch.evaluation.hybrid import hybrid_factory, reciprocal_rank_fusion
from euregsearch.provenance import ArticleRef


def ref(article, language="en"):
    return ArticleRef(celex="32014L0065", article=article, language=language,
                      version="consolidated", retrieved="2026-08-21", text=f"a{article}")


def test_fusion_rewards_agreement_between_runs():
    a, b, c = ref("1"), ref("2"), ref("3")
    fused = reciprocal_rank_fusion([[(a, 9.0), (b, 1.0)], [(a, 0.9), (c, 0.1)]], k=3)
    assert fused[0][0].article == "1"


def test_fusion_keeps_items_found_by_only_one_run():
    a, b = ref("1"), ref("2")
    fused = reciprocal_rank_fusion([[(a, 1.0)], [(b, 1.0)]], k=5)
    assert {r.article for r, _ in fused} == {"1", "2"}


def test_fusion_respects_k():
    runs = [[(ref(str(i)), 1.0) for i in range(10)]]
    assert len(reciprocal_rank_fusion(runs, k=3)) == 3


def test_fusion_distinguishes_languages():
    fused = reciprocal_rank_fusion([[(ref("1", "en"), 1.0)], [(ref("1", "nl"), 1.0)]], k=5)
    assert len(fused) == 2


def test_fusion_of_nothing_is_empty():
    assert reciprocal_rank_fusion([[], []], k=5) == []


def test_hybrid_factory_queries_both_retrievers():
    calls = []

    def lex(language):
        return lambda q, k: (calls.append("lex"), [(ref("1"), 1.0)])[1]

    def dense(language):
        return lambda q, k: (calls.append("dense"), [(ref("2"), 1.0)])[1]

    hits = hybrid_factory(lex, dense)("en")("query", k=5)
    assert calls == ["lex", "dense"]
    assert len(hits) == 2
