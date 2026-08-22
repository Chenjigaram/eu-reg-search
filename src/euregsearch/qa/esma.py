from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .citations import parse_citations

MINIMUM_PAIRS = 120

QUESTION = re.compile(r"^\s*Question\s+(\d+)[^\n]*$", re.IGNORECASE | re.MULTILINE)
ANSWER = re.compile(r"^\s*Answer\s+(\d+)[^\n]*$", re.IGNORECASE | re.MULTILINE)

# Joint Committee PRIIPs Q&As use "N. <question>" with the answer following
# immediately and no answer marker. The question ends at its final question mark.
NUMBERED = re.compile(r"^\s*(\d{1,3})\.\s+(?=[A-Z\"'(])", re.MULTILINE)


class QAEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str
    answer: str
    language: str
    source: str
    citations: list[tuple[str, str]]


def _blocks(text: str) -> list[tuple[str, str]]:
    marks = [(m.start(), m.end(), "q") for m in QUESTION.finditer(text)]
    marks += [(m.start(), m.end(), "a") for m in ANSWER.finditer(text)]
    marks.sort()
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for index, (_start, end, kind) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        body = text[end:stop].strip()
        if kind == "q":
            pending = body
        elif pending is not None:
            pairs.append((pending, body))
            pending = None
    return pairs


def _numbered_blocks(text: str) -> list[tuple[str, str]]:
    marks = [(m.start(), m.end()) for m in NUMBERED.finditer(text)]
    pairs: list[tuple[str, str]] = []
    for index, (_start, end) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        block = text[end:stop].strip()
        cut = block.rfind("?")
        if cut == -1 or cut == len(block) - 1:
            continue
        question, answer = block[: cut + 1].strip(), block[cut + 1 :].strip()
        if question and answer:
            pairs.append((question, answer))
    return pairs


def extract_entries(pages: list[str], language: str, source: str) -> list[QAEntry]:
    text = "\n".join(pages)
    blocks = _blocks(text) or _numbered_blocks(text)
    entries: list[QAEntry] = []
    for question, answer in blocks:
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
