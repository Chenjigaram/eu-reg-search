from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

LANGUAGES = ("en", "nl", "de", "fr")

EURLEX_ATTRIBUTION = (
    "Source: EUR-Lex, © European Union. Reused under Commission Decision 2011/833/EU; "
    "consolidated texts and editorial content are licensed CC BY 4.0."
)

BASE_URL = "https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/?uri=CELEX:{celex}"


class ArticleRef(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    celex: str
    article: str
    language: Literal["en", "nl", "de", "fr"]
    version: Literal["original", "consolidated"]
    retrieved: str
    anchor: str | None = None
    text: str = ""

    @field_validator("celex", "article", "retrieved")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    def deep_link(self) -> str:
        url = BASE_URL.format(lang=self.language.upper(), celex=self.celex)
        return f"{url}#{self.anchor}" if self.anchor else url

    def is_complete(self) -> bool:
        return all([self.celex, self.article, self.language, self.version, self.retrieved, self.deep_link()])

    def key(self) -> tuple[str, str]:
        return (self.celex, self.article)
