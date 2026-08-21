from __future__ import annotations

from ..index.embedder import Embedder
from ..index.store import VectorStore
from ..provenance import ArticleRef


def dense_search_fn(embedder: Embedder, store: VectorStore):
    def search(query: str, k: int = 10) -> list[tuple[ArticleRef, float]]:
        return store.search(embedder.encode([query], is_query=True)[0], k)

    return search


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
