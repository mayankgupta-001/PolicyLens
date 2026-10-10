from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class PolicyEmbedder:
    """
    Creates semantic embeddings for PolicyLens policy clauses
    and user queries.
    """

    MODEL_NAME = "BAAI/bge-m3"
    EMBEDDING_DIMENSION = 1024

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 16,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def _validate_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        text = text.strip()

        if not text:
            raise ValueError("Text cannot be empty.")

        return text

    def embed_documents(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        """
        Embed multiple policy clauses.
        """

        texts = [
            self._validate_text(text)
            for text in texts
        ]

        if not texts:
            return np.empty(
                (0, self.EMBEDDING_DIMENSION),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self._validate_dimension(embeddings)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query.
        """

        query = self._validate_text(query)

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        embedding = embedding.reshape(1, -1)

        self._validate_dimension(embedding)

        return embedding[0]

    def embed_queries(
        self,
        queries: Iterable[str],
    ) -> np.ndarray:
        """
        Embed multiple user queries.
        """

        queries = [
            self._validate_text(query)
            for query in queries
        ]

        if not queries:
            return np.empty(
                (0, self.EMBEDDING_DIMENSION),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self._validate_dimension(embeddings)

        return embeddings

    def _validate_dimension(
        self,
        embeddings: np.ndarray,
    ) -> None:

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got {embeddings.shape}"
            )

        if embeddings.shape[1] != self.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {self.EMBEDDING_DIMENSION} dimensions, "
                f"got {embeddings.shape[1]}"
            )

    @property
    def dimension(self) -> int:
        return self.EMBEDDING_DIMENSION


if __name__ == "__main__":

    print("Loading embedding model...")

    embedder = PolicyEmbedder()

    documents = [
        "Hospitalization expenses are covered under this policy.",
        "Pre-existing diseases are subject to a waiting period.",
    ]

    print("Generating document embeddings...")

    document_embeddings = embedder.embed_documents(
        documents
    )

    print(
        "Document embedding shape:",
        document_embeddings.shape
    )

    query = "Is hospitalization covered?"

    print("Generating query embedding...")

    query_embedding = embedder.embed_query(query)

    print(
        "Query embedding shape:",
        query_embedding.shape
    )

    print("\nEmbedding test successful.")