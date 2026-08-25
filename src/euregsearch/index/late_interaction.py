from __future__ import annotations

import numpy as np

from ..provenance import ArticleRef


def maxsim(query: np.ndarray, document: np.ndarray) -> float:
    """ColBERT scoring: each query token takes its best match in the document, then sum.

    Both matrices are (tokens, dimensions) and row-normalised, so the dot product is a
    cosine. Unlike mean pooling this never averages a provision away: one strongly
    matching sentence is enough for the tokens that matter.
    """
    if query.size == 0 or document.size == 0:
        return 0.0
    return float((query @ document.T).max(axis=1).sum())


class LateInteractionStore:
    """Documents kept as one packed token matrix so a query scores them in a single product.

    Scoring each document separately spends more time in Python than in arithmetic; packing
    them and taking a segmented maximum turns the whole corpus into one matmul.
    """

    def __init__(self) -> None:
        self.refs: list[ArticleRef] = []
        self._matrices: list[np.ndarray] = []
        self._packed: np.ndarray | None = None
        self._starts: np.ndarray | None = None

    def add(self, refs: list[ArticleRef], matrices: list[np.ndarray]) -> None:
        if len(refs) != len(matrices):
            raise ValueError("refs and matrices must line up")
        self.refs.extend(refs)
        self._matrices.extend(matrices)
        self._packed = None

    def __len__(self) -> int:
        return len(self.refs)

    def _pack(self) -> None:
        usable = [m for m in self._matrices if m.size]
        if not usable:
            self._packed = np.empty((0, 0), dtype=np.float32)
            self._starts = np.empty(0, dtype=np.int64)
            return
        lengths = [max(len(m), 1) for m in self._matrices]
        self._starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        filler = np.zeros((1, usable[0].shape[1]), dtype=np.float32)
        self._packed = np.vstack([m.astype(np.float32) if m.size else filler
                                  for m in self._matrices])
        self._matrices = []  # packed copy is authoritative; holding both doubles memory

    def search(self, query: np.ndarray, k: int) -> list[tuple[ArticleRef, float]]:
        if self._packed is None:
            self._pack()
        if self._packed.size == 0 or query.size == 0:
            return []
        similarity = query.astype(np.float32) @ self._packed.T
        per_document = np.maximum.reduceat(similarity, self._starts, axis=1).sum(axis=0)
        top = np.argsort(-per_document)[:k]
        return [(self.refs[i], float(per_document[i])) for i in top]


def encode_tokens(model, texts: list[str], is_query: bool = False,
                  batch_size: int = 8) -> list[np.ndarray]:
    """Row-normalised token embeddings per text, padding removed.

    Same encoder and same prefixes as the pooled path, so a comparison between the two
    isolates the aggregation and nothing else.
    """
    import torch

    prefix = "query: " if is_query else "passage: "
    matrices: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start:start + batch_size]]
        features = model.tokenize(batch)
        with torch.no_grad():
            features = model[0](features)
        embeddings = torch.nn.functional.normalize(features["token_embeddings"], dim=-1)
        mask = features["attention_mask"].bool()
        for row in range(len(batch)):
            matrices.append(embeddings[row][mask[row]].cpu().numpy().astype(np.float16))
    return matrices
