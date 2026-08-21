from __future__ import annotations

import math
from collections.abc import Iterable

Pair = tuple[str, str]


def _cut(retrieved: Iterable[Pair], k: int) -> list[Pair]:
    return list(retrieved)[:k]


def recall_at_k(retrieved: Iterable[Pair], relevant: Iterable[Pair], k: int) -> float:
    gold = set(relevant)
    if not gold:
        return 0.0
    found = sum(1 for pair in _cut(retrieved, k) if pair in gold)
    return found / len(gold)


def mrr_at_k(retrieved: Iterable[Pair], relevant: Iterable[Pair], k: int) -> float:
    gold = set(relevant)
    for position, pair in enumerate(_cut(retrieved, k), start=1):
        if pair in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Iterable[Pair], relevant: Iterable[Pair], k: int) -> float:
    gold = set(relevant)
    if not gold:
        return 0.0
    gain = sum(1.0 / math.log2(i + 1) for i, pair in enumerate(_cut(retrieved, k), start=1) if pair in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return gain / ideal if ideal else 0.0
