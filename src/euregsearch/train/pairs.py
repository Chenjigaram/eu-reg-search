from __future__ import annotations

import random
import re
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


def build_cross_lingual_pairs(refs: list[ArticleRef], per_article: int | None = None,
                              seed: int = 42) -> list[Pair]:
    """Align the same provision across languages.

    With four languages this yields up to ten ordered directions per article, which is why
    an unbounded build drowns out question-to-passage signal. `per_article` caps it.
    """
    rng = random.Random(seed)
    pairs: list[Pair] = []
    for key, by_language in _grouped(refs).items():
        directions = [d for d in permutations(by_language, 2) if d not in HELD_OUT_DIRECTIONS]
        if per_article is not None and len(directions) > per_article:
            directions = rng.sample(directions, per_article)
        for source, target in directions:
            pairs.append((by_language[source].text, by_language[target].text, key))
    return pairs


SENTENCE = re.compile(r"(?<=[.;:])\s+")


def build_ict_pairs(refs: list[ArticleRef], exclude: set[tuple[str, str]] | None = None,
                    per_article: int = 2, min_words: int = 8, seed: int = 42) -> list[Pair]:
    """Inverse Cloze Task pairs: a sentence drawn from the article is the query, the rest is
    the passage. This teaches question-to-passage matching, which formulaic templates
    ("What does Article N provide?") do not -- they teach the model to match article numbers.
    """
    rng = random.Random(seed)
    blocked = exclude or set()
    pairs: list[Pair] = []
    for key, by_language in _grouped(refs).items():
        if key in blocked:
            continue
        for ref in by_language.values():
            sentences = [s.strip() for s in SENTENCE.split(ref.text) if len(s.split()) >= min_words]
            if len(sentences) < 2:
                continue
            for chosen in rng.sample(sentences, min(per_article, len(sentences))):
                remainder = " ".join(s for s in sentences if s != chosen)
                if remainder:
                    pairs.append((chosen, remainder, key))
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


def build_passage_ict_pairs(refs: list[ArticleRef], exclude: set[tuple[str, str]] | None = None,
                            per_passage: int = 1, min_words: int = 8, seed: int = 42) -> list[Pair]:
    """ICT pairs over passages rather than whole articles, so a positive is exactly the unit
    the index serves. Training on whole articles truncates them to the training window while
    retrieval scores chunks, and the model never sees the text it is later asked to match.
    """
    rng = random.Random(seed)
    blocked = exclude or set()
    pairs: list[Pair] = []
    for ref in refs:
        if ref.celex == HELD_OUT_INSTRUMENT or ref.key() in blocked:
            continue
        sentences = [s.strip() for s in SENTENCE.split(ref.text) if len(s.split()) >= min_words]
        if len(sentences) < 2:
            continue
        for chosen in rng.sample(sentences, min(per_passage, len(sentences))):
            remainder = " ".join(s for s in sentences if s != chosen)
            if remainder:
                pairs.append((chosen, remainder, ref.key()))
    return pairs
