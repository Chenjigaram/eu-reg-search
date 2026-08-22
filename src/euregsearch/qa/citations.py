from __future__ import annotations

import re

INSTRUMENT_ALIASES = {
    "2014/65": "32014L0065",
    "600/2014": "32014R0600",
    "1286/2014": "32014R1286",
    "2017/653": "32017R0653",
    "2017/565": "32017R0565",
}

NAME_ALIASES = {
    "mifid": "32014L0065",
    "mifir": "32014R0600",
    "priips": "32014R1286",
}

CITATION = re.compile(
    r"Article\s+(?P<article>\d+[a-z]?)"
    r"(?:\([^)]*\))*"
    r"\s+of\s+(?P<instrument>[^,;]{3,80}?)(?=\s+and\s+Article\b|\s*[,;]|\.\s|$)",
    re.IGNORECASE,
)

NUMBER = re.compile(r"\d{3,4}/\d{2,4}")


def _resolve(instrument: str) -> str | None:
    # Numbers are more specific than names: "MiFID II Delegated Regulation (EU) 2017/565"
    # must resolve to the delegated regulation, not to MiFID II itself.
    for match in NUMBER.finditer(instrument):
        celex = INSTRUMENT_ALIASES.get(match.group(0))
        if celex:
            return celex
    lowered = instrument.lower().strip()
    for alias, celex in NAME_ALIASES.items():
        if alias in lowered:
            return celex
    return None


def parse_citations(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for match in CITATION.finditer(text or ""):
        celex = _resolve(match.group("instrument"))
        if celex is None:
            continue
        pair = (celex, match.group("article").lower())
        if pair not in found:
            found.append(pair)
    return found
