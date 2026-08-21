from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .citations import parse_citations

MINIMUM_PAIRS = 120

QUESTION = re.compile(r"^\s*Question\s+(\d+)[^\n]*$", re.IGNORECASE | re.MULTILINE)
ANSWER = re.compile(r"^\s*Answer\s+(\d+)[^\n]*$", re.IGNORECASE | re.MULTILINE)


class QAEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str
    answer: str
    language: str
    source: str
    citations: list[tuple[str, str]]


def _blocks(text: str) -> list[tuple[str, str]]:
    marks = [(m.start(), m.end(), "q", m.group(1)) for m in QUESTION.finditer(text)]
    marks += [(m.start(), m.end(), "a", m.group(1)) for m in ANSWER.finditer(text)]
    marks.sort()
    pairs: dict[str, dict[str, str]] = {}
    for index, (_start, end, kind, number) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        pairs.setdefault(number, {})[kind] = text[end:stop].strip()
    return [(v["q"], v["a"]) for _k, v in sorted(pairs.items()) if "q" in v and "a" in v]


def extract_entries(pages: list[str], language: str, source: str) -> list[QAEntry]:
    entries: list[QAEntry] = []
    for question, answer in _blocks("\n".join(pages)):
        citations = parse_citations(answer)
        if not citations:
            continue
        entries.append(QAEntry(question=question, answer=answer, language=language,
                               source=source, citations=citations))
    return entries


def read_pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def check_abort_gate(entries: list[QAEntry]) -> tuple[int, bool]:
    usable = sum(1 for entry in entries if entry.citations)
    return usable, usable >= MINIMUM_PAIRS
