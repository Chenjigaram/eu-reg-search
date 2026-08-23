from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, dimensions: int | None = None,
                 threads: int | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if threads:
            torch.set_num_threads(threads)
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dimensions = dimensions

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        vectors = self.model.encode([prefix + t for t in texts], convert_to_numpy=True,
                                    normalize_embeddings=True, batch_size=16)
        if self.dimensions:
            from .store import truncate_and_normalise

            vectors = truncate_and_normalise(vectors, self.dimensions)
        return vectors
