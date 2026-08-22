from __future__ import annotations

from collections.abc import Callable

from ..provenance import ArticleRef

SearchFn = Callable[[str, int], list[tuple[ArticleRef, float]]]

RRF_K = 60


def reciprocal_rank_fusion(runs: list[list[tuple[ArticleRef, float]]], k: int,
                           rrf_k: int = RRF_K) -> list[tuple[ArticleRef, float]]:
    """Combine ranked lists by reciprocal rank.

    Rank-based rather than score-based because BM25 scores and cosine similarities live on
    different scales and cannot be added directly. Each system contributes 1/(rrf_k + rank).
    """
    scores: dict[tuple[str, str, str], float] = {}
    seen: dict[tuple[str, str, str], ArticleRef] = {}
    for run in runs:
        for rank, (ref, _score) in enumerate(run, start=1):
            identity = (ref.celex, ref.article, ref.language)
            seen.setdefault(identity, ref)
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: -item[1])[:k]
    return [(seen[identity], score) for identity, score in ordered]


def hybrid_factory(lexical_factory, dense_factory, depth: int = 50):
    """Fuse a lexical and a dense retriever, per target language."""

    def factory(language: str) -> SearchFn:
        lexical = lexical_factory(language)
        dense = dense_factory(language)

        def search(query: str, k: int = 10) -> list[tuple[ArticleRef, float]]:
            return reciprocal_rank_fusion([lexical(query, depth), dense(query, depth)], k)

        return search

    return factory
