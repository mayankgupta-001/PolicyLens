from __future__ import annotations

import json
from pathlib import Path

import faiss

from .embeddings import PolicyEmbedder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INDEX_FILE = DATA_DIR / "policy_index.faiss"
METADATA_FILE = DATA_DIR / "metadata.json"


class PolicyRetriever:
    """
    Semantic retriever for PolicyLens insurance policy clauses.

    Uses:
        BGE-M3 embeddings
        FAISS inner-product similarity
        Metadata stored alongside the FAISS index
    """

    def __init__(
        self,
        index_path: Path = INDEX_FILE,
        metadata_path: Path = METADATA_FILE,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        # ---------------------------------------------------------
        # Validate files
        # ---------------------------------------------------------

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found:\n"
                f"{self.index_path}\n\n"
                f"Run build_index.py first."
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found:\n"
                f"{self.metadata_path}\n\n"
                f"Run build_index.py first."
            )

        # ---------------------------------------------------------
        # Load FAISS index
        # ---------------------------------------------------------

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(self.index_path)
        )

        # ---------------------------------------------------------
        # Load metadata
        # ---------------------------------------------------------

        print("Loading metadata...")

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        # ---------------------------------------------------------
        # Validate index and metadata
        # ---------------------------------------------------------

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata are out of sync.\n"
                f"FAISS vectors: {self.index.ntotal}\n"
                f"Metadata records: {len(self.metadata)}"
            )

        # ---------------------------------------------------------
        # Load embedding model
        # ---------------------------------------------------------

        print("Loading embedding model...")

        self.embedder = PolicyEmbedder()

        # ---------------------------------------------------------
        # Validate embedding dimension
        # ---------------------------------------------------------

        if self.index.d != self.embedder.dimension:
            raise ValueError(
                "Embedding dimension mismatch.\n"
                f"FAISS dimension: {self.index.d}\n"
                f"Model dimension: {self.embedder.dimension}"
            )

        print(
            f"Retriever ready: {self.index.ntotal} clauses"
        )

    # =============================================================
    # SEARCH
    # =============================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        policy_id: str | None = None,
    ) -> list[dict]:
        """
        Search for the most semantically relevant clauses.

        Args:
            query:
                User's natural-language question.

            top_k:
                Number of final results to return.

            policy_id:
                Optional policy filter.

        Returns:
            List of retrieved clause dictionaries.
        """

        # ---------------------------------------------------------
        # Validate query
        # ---------------------------------------------------------

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # ---------------------------------------------------------
        # Create query embedding
        # ---------------------------------------------------------

        query_embedding = self.embedder.embed_query(
            query
        )

        query_embedding = query_embedding.reshape(
            1,
            -1
        )

        # ---------------------------------------------------------
        # Retrieve a larger candidate pool
        #
        # Instead of retrieving only top_k directly,
        # retrieve up to 20 candidates first.
        #
        # This gives us a better candidate pool, especially
        # when a policy filter is applied.
        # ---------------------------------------------------------

        search_k = min(
            max(top_k * 4, 20),
            self.index.ntotal
        )

        # ---------------------------------------------------------
        # FAISS semantic search
        # ---------------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            search_k
        )

        results = []

        # ---------------------------------------------------------
        # Process candidates
        # ---------------------------------------------------------

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            if index_id == -1:
                continue

            metadata = self.metadata[index_id]

            # -----------------------------------------------------
            # Optional policy filtering
            # -----------------------------------------------------

            if (
                policy_id is not None
                and metadata["policy_id"] != policy_id
            ):
                continue

            # -----------------------------------------------------
            # Create result
            # -----------------------------------------------------

            result = {
                "score": float(score),
                "rank": len(results) + 1,
                "clause_id": metadata["clause_id"],
                "policy_id": metadata["policy_id"],
                "page_start": metadata["page_start"],
                "page_end": metadata["page_end"],
                "section": metadata["section"],
                "subsection": metadata["subsection"],
                "marker": metadata["marker"],
                "clause_text": metadata["clause_text"],
            }

            results.append(result)

            # -----------------------------------------------------
            # Stop after collecting requested number of results
            # -----------------------------------------------------

            if len(results) >= top_k:
                break

        return results

    # =============================================================
    # PRINT RESULTS
    # =============================================================

    @staticmethod
    def print_results(
        query: str,
        results: list[dict],
    ) -> None:

        print("\n")
        print("=" * 80)
        print("RETRIEVAL RESULTS")
        print("=" * 80)

        print(f"\nQuery:")
        print(query)

        print(
            f"\nResults found: {len(results)}"
        )

        for result in results:

            print("\n" + "-" * 80)

            print(
                f"Rank:       {result['rank']}"
            )

            print(
                f"Score:      {result['score']:.4f}"
            )

            print(
                f"Clause ID:  {result['clause_id']}"
            )

            print(
                f"Policy ID:  {result['policy_id']}"
            )

            print(
                f"Page:       "
                f"{result['page_start']}"
                f"-"
                f"{result['page_end']}"
            )

            print(
                f"Section:    {result['section']}"
            )

            if result["subsection"]:
                print(
                    f"Subsection: {result['subsection']}"
                )

            if result["marker"]:
                print(
                    f"Marker:     {result['marker']}"
                )

            print("\nClause:")

            print(
                result["clause_text"]
            )

        print("\n" + "=" * 80)

    # =============================================================
    # INTERACTIVE SEARCH
    # =============================================================

    def interactive_search(self):

        print("\n")
        print("=" * 80)
        print("POLICYLENS SEMANTIC RETRIEVAL")
        print("=" * 80)

        print("\nType an insurance question.")
        print("Type 'exit' to quit.")

        while True:

            try:
                query = input(
                    "\nQuestion: "
                ).strip()

            except KeyboardInterrupt:

                print(
                    "\nExiting."
                )

                break

            if query.lower() in {
                "exit",
                "quit",
            }:

                print(
                    "Exiting."
                )

                break

            if not query:
                continue

            try:

                results = self.search(
                    query=query,
                    top_k=5,
                )

                self.print_results(
                    query,
                    results,
                )

            except Exception as error:

                print(
                    f"\nRetrieval error: {error}"
                )


# =============================================================
# MAIN
# =============================================================

def interactive_search():

    retriever = PolicyRetriever()

    retriever.interactive_search()


if __name__ == "__main__":
    interactive_search()