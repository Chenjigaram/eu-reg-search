from __future__ import annotations

from ..index.chunking import chunk_refs, collapse_by_article
from ..index.embedder import Embedder
from ..index.late_interaction import LateInteractionStore, encode_tokens
from ..provenance import ArticleRef

COLLAPSE_DEPTH = 5


def late_factory(embedder: Embedder, refs: list[ArticleRef], chunked: bool = True):
    """Retrievers scoring by MaxSim over token embeddings rather than a pooled vector.

    Same encoder and prefixes as the dense path, so the only thing that differs is how
    the token embeddings are aggregated.
    """
    by_language: dict[str, list[ArticleRef]] = {}
    for ref in refs:
        by_language.setdefault(ref.language, []).append(ref)
    built: dict[str, object] = {}

    def factory(language: str):
        if language not in built:
            subset = by_language.get(language, [])
            if chunked and subset:
                subset = chunk_refs(subset, embedder.model.tokenizer)
            store = LateInteractionStore()
            if subset:
                store.add(subset, encode_tokens(embedder.model, [r.text for r in subset]))

            def search(query: str, k: int = 10):
                vector = encode_tokens(embedder.model, [query], is_query=True)[0]
                if not chunked:
                    return store.search(vector, k)
                return collapse_by_article(store.search(vector, k * COLLAPSE_DEPTH), k)

            built[language] = search
        return built[language]

    return factory
