from __future__ import annotations

from ..index.chunking import chunk_refs, collapse_by_article
from ..index.embedder import Embedder
from ..index.store import VectorStore
from ..provenance import ArticleRef

COLLAPSE_DEPTH = 5


def dense_search_fn(embedder: Embedder, store: VectorStore, chunked: bool = False):
    def search(query: str, k: int = 10) -> list[tuple[ArticleRef, float]]:
        vector = embedder.encode([query], is_query=True)[0]
        if not chunked:
            return store.search(vector, k)
        return collapse_by_article(store.search(vector, k * COLLAPSE_DEPTH), k)

    return search


def dense_factory(embedder: Embedder, refs: list[ArticleRef], chunked: bool = True):
    """Encode once per language and return a factory of language-restricted retrievers."""
    by_language: dict[str, list[ArticleRef]] = {}
    for ref in refs:
        by_language.setdefault(ref.language, []).append(ref)
    built: dict[str, object] = {}

    def factory(language: str):
        if language not in built:
            subset = by_language.get(language, [])
            if chunked and subset:
                subset = chunk_refs(subset, embedder.model.tokenizer)
            store = VectorStore()
            if subset:
                store.add(subset, embedder.encode([r.text for r in subset], is_query=False))
            built[language] = dense_search_fn(embedder, store, chunked)
        return built[language]

    return factory


def comparison_table(summaries: list[dict]) -> str:
    if not summaries:
        return "No results to report."
    rows = ["| System | Queries | nDCG@10 | Recall@10 | MRR@10 | Provenance |",
            "| --- | --- | --- | --- | --- | --- |"]
    for s in sorted(summaries, key=lambda x: -x["ndcg"]):
        provenance = "complete" if s["provenance_complete"] else "**DEFECT**"
        rows.append(f"| {s['system']} | {s['queries']} | {s['ndcg']:.3f} | "
                    f"{s['recall']:.3f} | {s['mrr']:.3f} | {provenance} |")
    return "\n".join(rows)


def slice_table(summary: dict) -> str:
    slices = summary.get("slices") or {}
    if not slices:
        return f"{summary['system']}: no slices recorded."
    rows = ["| Slice | Queries | nDCG@10 | Recall@10 | MRR@10 |", "| --- | --- | --- | --- | --- |"]
    for name, values in sorted(slices.items()):
        rows.append(f"| {name} | {values['queries']} | {values['ndcg']:.3f} | "
                    f"{values['recall']:.3f} | {values['mrr']:.3f} |")
    return "\n".join(rows)
