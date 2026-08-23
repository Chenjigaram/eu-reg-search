from __future__ import annotations

import numpy as np

from ..provenance import ArticleRef


def truncate_and_normalise(vectors: np.ndarray, dimensions: int) -> np.ndarray:
    reduced = np.asarray(vectors, dtype=np.float32)[:, :dimensions]
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return reduced / norms


class VectorStore:
    def __init__(self) -> None:
        self.refs: list[ArticleRef] = []
        self.vectors: np.ndarray | None = None

    def add(self, refs: list[ArticleRef], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if len(refs) != matrix.shape[0]:
            raise ValueError(f"{len(refs)} refs but {matrix.shape[0]} vectors")
        self.refs.extend(refs)
        self.vectors = matrix if self.vectors is None else np.vstack([self.vectors, matrix])

    def __len__(self) -> int:
        return len(self.refs)

    def search(self, vector: np.ndarray, k: int = 10) -> list[tuple[ArticleRef, float]]:
        if self.vectors is None or not self.refs:
            return []
        query = np.asarray(vector, dtype=np.float32)
        query = query / (np.linalg.norm(query) or 1.0)
        matrix = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-12)
        scores = matrix @ query
        order = np.argsort(-scores)[:k]
        return [(self.refs[i], float(scores[i])) for i in order]
