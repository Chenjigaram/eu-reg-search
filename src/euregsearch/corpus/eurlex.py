from __future__ import annotations

import re

from bs4 import BeautifulSoup

from ..provenance import ArticleRef

ARTICLE_CLASS = "oj-ti-art"
ARTICLE_NUMBER = re.compile(r"Art(?:icle|ikel|\.)?\s*([0-9]+[a-z]?)", re.IGNORECASE)


def _article_number(heading: str) -> str | None:
    match = ARTICLE_NUMBER.search(heading)
    return match.group(1) if match else None


def segment_articles(html: str, celex: str, language: str, version: str, retrieved: str) -> list[ArticleRef]:
    soup = BeautifulSoup(html, "html.parser")
    headings = soup.find_all("p", class_=ARTICLE_CLASS)
    refs: list[ArticleRef] = []

    for heading in headings:
        number = _article_number(heading.get_text(" ", strip=True))
        if number is None:
            continue
        body: list[str] = []
        for node in heading.find_next_siblings():
            classes = node.get("class") or []
            if ARTICLE_CLASS in classes:
                break
            text = node.get_text(" ", strip=True)
            if text:
                body.append(text)
        refs.append(
            ArticleRef(
                celex=celex,
                article=number,
                language=language,
                version=version,
                retrieved=retrieved,
                anchor=heading.get("id"),
                text=" ".join(body),
            )
        )
    return refs
