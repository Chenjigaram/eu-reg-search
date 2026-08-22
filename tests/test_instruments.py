import json

import pytest

from euregsearch.corpus.instruments import INSTRUMENTS, LANGUAGES, build_corpus, fetch_html

SAMPLE = """<html><body>
<p class="oj-ti-art" id="d1e1-1">Article 1</p><p>Scope.</p>
</body></html>"""


def test_all_five_instruments_are_registered():
    assert {i.celex for i in INSTRUMENTS} == {
        "32014L0065", "32014R0600", "32014R1286", "32017R0653", "32017R0565"}


def test_exactly_four_languages():
    assert LANGUAGES == ("en", "nl", "de", "fr")


def test_every_instrument_has_a_short_name():
    assert all(i.short_name for i in INSTRUMENTS)


def test_fetch_uses_the_cache_when_present(tmp_path):
    cached = tmp_path / "32014L0065.en.html"
    cached.write_text(SAMPLE, encoding="utf-8")
    assert fetch_html("32014L0065", "en", tmp_path) == SAMPLE


def test_fetch_without_cache_and_without_network_raises(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr("euregsearch.corpus.instruments.requests.get", refuse)
    with pytest.raises(RuntimeError):
        fetch_html("32014L0065", "en", tmp_path)


def test_build_corpus_writes_one_record_per_article(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for instrument in INSTRUMENTS:
        for language in LANGUAGES:
            (cache / f"{instrument.celex}.{language}.html").write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / "articles.jsonl"
    count = build_corpus(cache, out, retrieved="2026-08-21")
    assert count == len(INSTRUMENTS) * len(LANGUAGES)
    first = json.loads(out.read_text().splitlines()[0])
    assert first["celex"] and first["article"] and first["language"]


def test_language_codes_map_to_iso_639_3():
    from euregsearch.corpus.instruments import ISO_639_3

    assert ISO_639_3 == {"en": "eng", "nl": "nld", "de": "deu", "fr": "fra"}
    assert set(ISO_639_3) == set(LANGUAGES)


def test_empty_body_is_rejected_rather_than_cached(tmp_path, monkeypatch):
    class EmptyResponse:
        status_code = 202
        content = b""
        text = ""

        def raise_for_status(self):
            return None

    monkeypatch.setattr("euregsearch.corpus.instruments.requests.get", lambda *a, **k: EmptyResponse())
    with pytest.raises(RuntimeError, match="empty body"):
        fetch_html("32014L0065", "en", tmp_path)
    assert not (tmp_path / "32014L0065.en.html").exists()
