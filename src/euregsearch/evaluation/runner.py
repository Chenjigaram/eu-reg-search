from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from ..provenance import ArticleRef
from ..qa.judgements import Judgement
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k

SearchFn = Callable[[str, int], list[tuple[ArticleRef, float]]]
SearchFactory = Callable[[str], SearchFn]


@dataclass
class SystemResult:
    name: str
    queries: int
    ndcg: float
    recall: float
    mrr: float
    provenance_complete: bool
    per_slice: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)

    def by_slice(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, rows in self.per_slice.items():
            out[name] = {
                "queries": len(rows),
                "ndcg": round(statistics.mean(r[0] for r in rows), 4),
                "recall": round(statistics.mean(r[1] for r in rows), 4),
                "mrr": round(statistics.mean(r[2] for r in rows), 4),
            }
        return out

    def summary(self) -> dict:
        return {"system": self.name, "queries": self.queries, "ndcg": round(self.ndcg, 4),
                "recall": round(self.recall, 4), "mrr": round(self.mrr, 4),
                "provenance_complete": self.provenance_complete, "slices": self.by_slice()}


def evaluate_system(name: str, search_factory: SearchFactory, judgements: list[Judgement],
                    k: int = 10) -> SystemResult:
    """Score a system. The factory yields a retriever restricted to one target language,
    so a cross-lingual judgement is genuinely answered from that language's articles only."""
    per_slice: dict[str, list[tuple[float, float, float]]] = {}
    scores: list[tuple[float, float, float]] = []
    complete = True
    cache: dict[str, SearchFn] = {}

    for judgement in judgements:
        if judgement.target_language not in cache:
            cache[judgement.target_language] = search_factory(judgement.target_language)
        hits = cache[judgement.target_language](judgement.query, k)
        complete = complete and all(ref.is_complete() for ref, _ in hits)
        retrieved = [ref.key() for ref, _ in hits]
        row = (
            ndcg_at_k(retrieved, judgement.relevant, k),
            recall_at_k(retrieved, judgement.relevant, k),
            mrr_at_k(retrieved, judgement.relevant, k),
        )
        scores.append(row)
        per_slice.setdefault(judgement.slice_name, []).append(row)

    def mean(index: int) -> float:
        return statistics.mean(s[index] for s in scores) if scores else 0.0

    return SystemResult(name, len(judgements), mean(0), mean(1), mean(2), complete, per_slice)
