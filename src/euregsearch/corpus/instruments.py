from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .eurlex import segment_articles

LANGUAGES = ("en", "nl", "de", "fr")

# The EUR-Lex web UI sits behind an AWS WAF bot challenge and returns 202 with an empty
# body to automated clients. Cellar is the Publications Office's sanctioned
# machine-to-machine endpoint; it negotiates language via ISO 639-3 codes.
CELLAR_URL = "http://publications.europa.eu/resource/celex/{celex}"
ISO_639_3 = {"en": "eng", "nl": "nld", "de": "deu", "fr": "fra"}


@dataclass(frozen=True)
class Instrument:
    celex: str
    short_name: str
    full_name: str


INSTRUMENTS = (
    Instrument("32014L0065", "MiFID II", "Directive 2014/65/EU"),
    Instrument("32014R0600", "MiFIR", "Regulation (EU) No 600/2014"),
    Instrument("32014R1286", "PRIIPs", "Regulation (EU) No 1286/2014"),
    Instrument("32017R0653", "PRIIPs RTS", "Commission Delegated Regulation (EU) 2017/653"),
    Instrument("32017R0565", "MiFID II DR", "Commission Delegated Regulation (EU) 2017/565"),
)


def fetch_html(celex: str, language: str, cache_dir: Path, pause: float = 1.0) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{celex}.{language}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    response = requests.get(
        CELLAR_URL.format(celex=celex),
        timeout=90,
        headers={"Accept": "application/xhtml+xml", "Accept-Language": ISO_639_3[language]},
    )
    response.raise_for_status()
    if not response.content:
        raise RuntimeError(f"empty body for {celex}/{language} (status {response.status_code})")
    cached.write_text(response.text, encoding="utf-8")
    time.sleep(pause)
    return response.text


def build_corpus(cache_dir: Path, out_path: Path, retrieved: str, version: str = "consolidated") -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for instrument in INSTRUMENTS:
            for language in LANGUAGES:
                html = fetch_html(instrument.celex, language, cache_dir)
                for ref in segment_articles(html, instrument.celex, language, version, retrieved):
                    handle.write(json.dumps(ref.model_dump(), ensure_ascii=False) + "\n")
                    written += 1
    return written
