from euregsearch.index.chunking import chunk_refs, chunk_text, collapse_by_article
from euregsearch.provenance import ArticleRef


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def ref(article, text, language="en"):
    return ArticleRef(celex="32014L0065", article=article, language=language,
                      version="consolidated", retrieved="2026-08-21", anchor="d1e1", text=text)


def test_short_text_is_left_whole():
    assert chunk_text("a b c", FakeTokenizer(), size=10, overlap=2) == ["a b c"]


def test_long_text_is_split():
    text = " ".join(str(i) for i in range(30))
    pieces = chunk_text(text, FakeTokenizer(), size=10, overlap=2)
    assert len(pieces) > 1
    assert all(len(p.split()) <= 10 for p in pieces)


def test_chunks_overlap_so_boundaries_are_not_lost():
    text = " ".join(str(i) for i in range(30))
    pieces = chunk_text(text, FakeTokenizer(), size=10, overlap=4)
    assert pieces[0].split()[-4:] == pieces[1].split()[:4]


def test_every_token_survives_chunking():
    text = " ".join(str(i) for i in range(37))
    pieces = chunk_text(text, FakeTokenizer(), size=10, overlap=3)
    assert set(text.split()) == {t for p in pieces for t in p.split()}


def test_chunked_refs_keep_article_identity_and_provenance():
    long_text = " ".join(str(i) for i in range(40))
    chunks = chunk_refs([ref("4", long_text)], FakeTokenizer(), size=10, overlap=2)
    assert len(chunks) > 1
    assert {c.key() for c in chunks} == {("32014L0065", "4")}
    assert all(c.deep_link().endswith("#d1e1") for c in chunks)
    assert all(c.is_complete() for c in chunks)


def test_collapse_keeps_best_scoring_chunk_per_article():
    a1, a2, b = ref("4", "x"), ref("4", "y"), ref("5", "z")
    collapsed = collapse_by_article([(a1, 0.2), (b, 0.5), (a2, 0.9)], k=10)
    assert [r.article for r, _ in collapsed] == ["4", "5"]
    assert collapsed[0][1] == 0.9
    assert collapsed[0][0].text == "y"


def test_collapse_respects_k():
    hits = [(ref(str(i), "t"), 1.0 / (i + 1)) for i in range(8)]
    assert len(collapse_by_article(hits, k=3)) == 3


def test_collapse_separates_languages():
    collapsed = collapse_by_article([(ref("4", "x", "en"), 0.5), (ref("4", "x", "nl"), 0.4)], k=10)
    assert len(collapsed) == 2


def test_passage_ict_pairs_are_built_per_chunk_not_per_article():
    from euregsearch.train.pairs import build_passage_ict_pairs

    long_text = ". ".join(f"clause {i} of the provision applies to firms" for i in range(20))
    chunks = chunk_refs([ref("4", long_text)], FakeTokenizer(), size=30, overlap=5)
    assert len(chunks) > 1
    pairs = build_passage_ict_pairs(chunks, per_passage=1)
    assert 1 < len(pairs) <= len(chunks)
    assert all(key == ("32014L0065", "4") for _a, _p, key in pairs)


def test_passage_ict_pairs_exclude_blocked_articles():
    from euregsearch.train.pairs import build_passage_ict_pairs

    text = ". ".join(f"clause {i} of the provision applies to firms" for i in range(6))
    pairs = build_passage_ict_pairs([ref("4", text)], exclude={("32014L0065", "4")})
    assert pairs == []


def test_passage_ict_query_is_not_inside_its_own_positive():
    from euregsearch.train.pairs import build_passage_ict_pairs

    text = ". ".join(f"clause {i} of the provision applies to firms" for i in range(6))
    for anchor, positive, _key in build_passage_ict_pairs([ref("4", text)], per_passage=3):
        assert anchor not in positive
