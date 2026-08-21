from __future__ import annotations

from collections import defaultdict
from itertools import permutations

from ..provenance import ArticleRef
from ..qa.judgements import HELD_OUT_DIRECTIONS, HELD_OUT_INSTRUMENT

Pair = tuple[str, str, tuple[str, str]]


def _grouped(refs: list[ArticleRef]) -> dict[tuple[str, str], dict[str, ArticleRef]]:
    groups: dict[tuple[str, str], dict[str, ArticleRef]] = defaultdict(dict)
    for ref in refs:
        if ref.celex == HELD_OUT_INSTRUMENT:
            continue
        groups[ref.key()][ref.language] = ref
    return groups


def build_cross_lingual_pairs(refs: list[ArticleRef]) -> list[Pair]:
    pairs: list[Pair] = []
    for key, by_language in _grouped(refs).items():
        for source, target in permutations(by_language, 2):
            if (source, target) in HELD_OUT_DIRECTIONS:
                continue
            pairs.append((by_language[source].text, by_language[target].text, key))
    return pairs


def build_synthetic_pairs(refs: list[ArticleRef], exclude: set[tuple[str, str]] | None = None) -> list[Pair]:
    blocked = exclude or set()
    pairs: list[Pair] = []
    for key, by_language in _grouped(refs).items():
        if key in blocked:
            continue
        for _language, ref in by_language.items():
            question = f"What does Article {ref.article} of {ref.celex} provide?"
            pairs.append((question, ref.text, key))
    return pairs
