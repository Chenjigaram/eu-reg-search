from __future__ import annotations

import random
from collections import defaultdict

from ..provenance import ArticleRef
from ..qa.judgements import Judgement

Pair = tuple[str, str, tuple[str, str]]


def k_folds(items: list, k: int = 5, seed: int = 42) -> list[tuple[list, list]]:
    """Split items into k (train, test) pairs. Every item is tested exactly once."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    buckets: list[list] = [[] for _ in range(k)]
    for position, item in enumerate(shuffled):
        buckets[position % k].append(item)
    folds = []
    for index in range(k):
        test = buckets[index]
        train = [item for other, bucket in enumerate(buckets) if other != index for item in bucket]
        folds.append((train, test))
    return folds


def build_supervised_pairs(judgements: list[Judgement], refs: list[ArticleRef]) -> list[Pair]:
    """Real question-to-provision pairs, which is the supervision the unsupervised runs lacked.

    Every passage of a cited article becomes a positive, matching how retrieval scores an
    article by its best passage.
    """
    by_key: dict[tuple[str, str, str], list[ArticleRef]] = defaultdict(list)
    for ref in refs:
        by_key[(ref.celex, ref.article, ref.language)].append(ref)
    pairs: list[Pair] = []
    for judgement in judgements:
        for celex, article in judgement.relevant:
            for ref in by_key.get((celex, article, judgement.target_language), []):
                pairs.append((judgement.query, ref.text, (celex, article)))
    return pairs
