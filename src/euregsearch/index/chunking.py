from __future__ import annotations

from ..provenance import ArticleRef

CHUNK_TOKENS = 400
OVERLAP_TOKENS = 80
MIN_TAIL_TOKENS = 40


def chunk_text(text: str, tokenizer, size: int = CHUNK_TOKENS,
               overlap: int = OVERLAP_TOKENS) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= size:
        return [text]
    step = size - overlap
    min_tail = max(1, min(MIN_TAIL_TOKENS, size // 4))
    pieces = []
    for start in range(0, len(ids), step):
        window = ids[start:start + size]
        if len(window) < min_tail and pieces:
            break
        pieces.append(tokenizer.decode(window))
        if start + size >= len(ids):
            break
    return pieces


def chunk_refs(refs: list[ArticleRef], tokenizer, size: int = CHUNK_TOKENS,
               overlap: int = OVERLAP_TOKENS) -> list[ArticleRef]:
    chunks = []
    for ref in refs:
        for piece in chunk_text(ref.text, tokenizer, size, overlap):
            chunks.append(ref.model_copy(update={"text": piece}))
    return chunks


def collapse_by_article(hits: list[tuple[ArticleRef, float]], k: int) -> list[tuple[ArticleRef, float]]:
    best: dict[tuple[str, str, str], tuple[ArticleRef, float]] = {}
    for ref, score in hits:
        identity = (ref.celex, ref.article, ref.language)
        if identity not in best or score > best[identity][1]:
            best[identity] = (ref, score)
    return sorted(best.values(), key=lambda item: -item[1])[:k]
