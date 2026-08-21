from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .esma import QAEntry

HELD_OUT_INSTRUMENT = "32017R0653"
HELD_OUT_DIRECTIONS = (("nl", "en"), ("de", "fr"))


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    query_language: str
    target_language: str
    relevant: list[tuple[str, str]]
    slice_name: str


def _slice_for(entry: QAEntry, query_language: str, target_language: str) -> str:
    if any(celex == HELD_OUT_INSTRUMENT for celex, _ in entry.citations):
        return "held_out_instrument"
    if query_language != target_language:
        if (query_language, target_language) in HELD_OUT_DIRECTIONS:
            return "held_out_direction"
        return "cross_lingual"
    return "same_language"


def build_judgements(entries: list[QAEntry], target_languages: list[str]) -> list[Judgement]:
    judgements: list[Judgement] = []
    for entry in entries:
        if not entry.citations:
            continue
        for target in target_languages:
            judgements.append(
                Judgement(
                    query=entry.question,
                    query_language=entry.language,
                    target_language=target,
                    relevant=list(entry.citations),
                    slice_name=_slice_for(entry, entry.language, target),
                )
            )
    return judgements


def evaluation_articles(judgements: list[Judgement]) -> set[tuple[str, str]]:
    return {pair for judgement in judgements for pair in judgement.relevant}


def training_is_disjoint(judgements: list[Judgement], synthetic_pair_keys: list[tuple[str, str]]) -> bool:
    """No SYNTHETIC training question may describe an article used in evaluation.

    Evaluation articles must be present in the index -- they are what retrieval returns --
    so disjointness cannot mean excluding them from the corpus. The leak this guards
    against is narrower: a generated question naming an article we later score on.
    Cross-lingual pairs are exempt; they carry no question, only two language versions
    of the same provision.
    """
    return not evaluation_articles(judgements) & set(synthetic_pair_keys)
