from __future__ import annotations

import re

from ..provenance import ArticleRef


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, refs: list[ArticleRef]) -> None:
        from rank_bm25 import BM25Okapi

        self.refs = refs
        self.model = BM25Okapi([tokenize(r.text) for r in refs]) if refs else None

    def search(self, query: str, k: int = 10) -> list[tuple[ArticleRef, float]]:
        if not self.refs or self.model is None:
            return []
        scores = self.model.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.refs[i], float(scores[i])) for i in order]


def bm25_factory(refs: list[ArticleRef]):
    """Return a factory yielding a BM25 retriever over one language only."""
    by_language: dict[str, list[ArticleRef]] = {}
    for ref in refs:
        by_language.setdefault(ref.language, []).append(ref)
    built: dict[str, BM25Retriever] = {}

    def factory(language: str):
        if language not in built:
            built[language] = BM25Retriever(by_language.get(language, []))
        return built[language].search

    return factory
