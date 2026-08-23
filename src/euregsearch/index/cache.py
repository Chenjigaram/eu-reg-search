from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class VectorCache:
    """Content-addressed store for passage embeddings.

    Encoding the corpus dominates every experiment and is deterministic for a given model,
    so vectors are keyed by a digest of the model name and the passage text. Changing the
    corpus re-encodes only what changed; changing the model shares nothing.
    """

    def __init__(self, path: Path, model_name: str) -> None:
        self.path = Path(path)
        self.model_name = model_name
        self.vectors: dict[str, np.ndarray] = {}

    @classmethod
    def empty(cls, model_name: str) -> VectorCache:
        return cls(Path("index/vectors.npz"), model_name)

    def _key(self, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def split(self, texts: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        known: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for text in texts:
            vector = self.vectors.get(self._key(text))
            if vector is None:
                if text not in missing:
                    missing.append(text)
            else:
                known[text] = vector
        return known, missing

    def store(self, texts: list[str], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if len(texts) != matrix.shape[0]:
            raise ValueError(f"{len(texts)} texts but {matrix.shape[0]} vectors")
        for text, vector in zip(texts, matrix, strict=True):
            self.vectors[self._key(text)] = vector

    def load(self) -> None:
        if not self.path.exists():
            return
        with np.load(self.path) as handle:
            self.vectors = {key: handle[key] for key in handle.files}

    def save(self) -> None:
        if not self.vectors:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self.vectors)
