from euregsearch.evaluation.baselines import BM25Retriever
from euregsearch.evaluation.runner import evaluate_system
from euregsearch.provenance import ArticleRef
from euregsearch.qa.judgements import Judgement


def ref(celex, article, text):
    return ArticleRef(celex=celex, article=article, language="en", version="consolidated",
                      retrieved="2026-08-21", anchor="d1e1-1", text=text)


REFS = [
    ref("32014L0065", "25", "assessment of suitability and appropriateness for clients"),
    ref("32014R1286", "8", "content of the key information document costs"),
    ref("32017R0565", "3", "record keeping obligations for investment firms"),
]

JUDGEMENTS = [
    Judgement(query="when does an appropriateness assessment apply", query_language="en",
              target_language="en", relevant=[("32014L0065", "25")], slice_name="same_language"),
    Judgement(query="what goes in the key information document", query_language="en",
              target_language="en", relevant=[("32014R1286", "8")], slice_name="same_language"),
]


def test_bm25_ranks_the_lexically_matching_article_first():
    hits = BM25Retriever(REFS).search("appropriateness assessment", k=3)
    assert hits[0][0].article == "25"


def test_bm25_returns_at_most_k():
    assert len(BM25Retriever(REFS).search("investment", k=2)) == 2


def test_evaluate_system_reports_metrics():
    result = evaluate_system("bm25", BM25Retriever(REFS).search, JUDGEMENTS, k=10)
    assert 0.0 <= result.ndcg <= 1.0
    assert result.queries == 2


def test_results_are_broken_down_by_slice():
    result = evaluate_system("bm25", BM25Retriever(REFS).search, JUDGEMENTS, k=10)
    assert "same_language" in result.by_slice()


def test_provenance_completeness_is_checked():
    result = evaluate_system("bm25", BM25Retriever(REFS).search, JUDGEMENTS, k=10)
    assert result.provenance_complete is True


def test_a_system_returning_nothing_scores_zero_without_crashing():
    result = evaluate_system("empty", lambda q, k: [], JUDGEMENTS, k=10)
    assert result.ndcg == 0.0 and result.recall == 0.0
