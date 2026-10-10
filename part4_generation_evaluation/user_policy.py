from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np

# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Reuse Part 1 ingestion
# -------------------------------------------------------------------

from part1_preprocessing.src.ingestion.pdf_extractor import extract_pdf
from part1_preprocessing.src.ingestion.clause_extractor import extract_clauses

# Reuse Part 3 embedding model
from part3_retrieval_rag.embeddings import PolicyEmbedder


# -------------------------------------------------------------------
# UserPolicy
# -------------------------------------------------------------------

class UserPolicy:
    """
    Ingest a user-provided insurance PDF using the same preprocessing
    pipeline used by Part 1.

    Pipeline:

        User PDF
            ↓
        Part 1 PDF extraction
            ↓
        Part 1 clause extraction
            ↓
        BGE-M3 embeddings
            ↓
        Temporary FAISS index
            ↓
        Semantic retrieval
    """

    def __init__(
        self,
        pdf_path: str | Path,
        enable_ocr: bool = True,
        top_k: int = 5,
    ):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found:\n{self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got: {self.pdf_path.suffix}"
            )

        self.enable_ocr = enable_ocr
        self.default_top_k = top_k

        # Create a stable user-policy ID from the filename.
        self.policy_id = self._create_policy_id(
            self.pdf_path.stem
        )

        # User-specific storage.
        self.user_data_dir = (
            PROJECT_ROOT
            / "part4_generation_evaluation"
            / "user_data"
            / self.policy_id
        )

        self.user_data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = (
            self.user_data_dir / "policy_index.faiss"
        )

        self.metadata_path = (
            self.user_data_dir / "metadata.json"
        )

        self.embedder = PolicyEmbedder()

        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self.clauses: list[dict[str, Any]] = []

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _create_policy_id(filename: str) -> str:
        """
        Convert a filename into a filesystem-safe policy ID.
        """

        clean_name = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            filename,
        ).strip("_")

        if not clean_name:
            clean_name = "uploaded_policy"

        return f"USER_{clean_name}"

    # ----------------------------------------------------------------
    # Part 1 ingestion
    # ----------------------------------------------------------------

    def extract_clauses(self) -> list[dict[str, Any]]:
        """
        Use Part 1's actual PDF extraction and clause extraction.
        """

        print("\n" + "=" * 80)
        print("PART 1 POLICY INGESTION")
        print("=" * 80)

        print(f"\nPDF:")
        print(self.pdf_path)

        print("\nExtracting PDF text...")

        extracted = extract_pdf(
            pdf_path=self.pdf_path,
            enable_ocr=self.enable_ocr,
        )

        print(
            f"Pages extracted: "
            f"{extracted['page_count']}"
        )

        print(
            f"Characters extracted: "
            f"{extracted['total_characters']}"
        )

        print(
            f"OCR available: "
            f"{extracted['ocr_available']}"
        )

        print("\nRunning Part 1 clause extraction...")

        clauses = extract_clauses(
            extracted=extracted,
            policy_id=self.policy_id,
        )

        if not clauses:
            raise ValueError(
                "Part 1 clause extraction produced zero clauses.\n"
                "The PDF may not contain extractable policy text."
            )

        self.clauses = clauses

        print(
            f"Clauses extracted: "
            f"{len(self.clauses)}"
        )

        return self.clauses

    # ----------------------------------------------------------------
    # Build FAISS index
    # ----------------------------------------------------------------

    def build_index(self) -> None:
        """
        Build a temporary FAISS index for this uploaded policy.
        """

        if not self.clauses:
            self.extract_clauses()

        print("\n" + "=" * 80)
        print("BUILDING USER POLICY INDEX")
        print("=" * 80)

        texts = [
            clause["clause_text"]
            for clause in self.clauses
            if clause.get("clause_text")
        ]

        if not texts:
            raise ValueError(
                "No usable clause text was extracted."
            )

        print(
            f"\nEmbedding {len(texts)} clauses "
            f"with BGE-M3..."
        )

        embeddings = self.embedder.embed_documents(
            texts
        )

        print(
            f"Embedding matrix shape: "
            f"{embeddings.shape}"
        )

        # BGE-M3 embeddings are normalized, so inner product
        # gives cosine similarity.
        self.index = faiss.IndexFlatIP(
            self.embedder.dimension
        )

        self.index.add(
            embeddings.astype(np.float32)
        )

        # Keep exactly the same metadata structure used
        # by the permanent Part 3 index.
        self.metadata = []

        for clause in self.clauses:
            self.metadata.append(
                {
                    "clause_id": clause["clause_id"],
                    "policy_id": clause["policy_id"],
                    "page_start": clause["page_start"],
                    "page_end": clause["page_end"],
                    "section": clause["section"],
                    "subsection": clause["subsection"],
                    "marker": clause["marker"],
                    "clause_text": clause["clause_text"],
                }
            )

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata are out of sync.\n"
                f"FAISS vectors: {self.index.ntotal}\n"
                f"Metadata records: {len(self.metadata)}"
            )

        # Save user-specific index.
        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        # Save user-specific metadata.
        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"\nFAISS vectors: "
            f"{self.index.ntotal}"
        )

        print(
            f"Metadata records: "
            f"{len(self.metadata)}"
        )

        print(
            f"\nIndex saved to:\n"
            f"{self.index_path}"
        )

        print(
            f"\nMetadata saved to:\n"
            f"{self.metadata_path}"
        )

        print("\nUSER POLICY INDEX BUILD SUCCESSFUL")

    # ----------------------------------------------------------------
    # Load existing user index
    # ----------------------------------------------------------------

    def load_index(self) -> None:
        """
        Load an already-created user-specific index.
        """

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"User policy index not found:\n"
                f"{self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"User policy metadata not found:\n"
                f"{self.metadata_path}"
            )

        print("Loading user FAISS index...")

        self.index = faiss.read_index(
            str(self.index_path)
        )

        print("Loading user metadata...")

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata are out of sync."
            )

        if self.index.d != self.embedder.dimension:
            raise ValueError(
                "Embedding dimension mismatch.\n"
                f"FAISS dimension: {self.index.d}\n"
                f"Model dimension: {self.embedder.dimension}"
            )

        print(
            f"User policy index loaded: "
            f"{self.index.ntotal} clauses"
        )

    # ----------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over the uploaded policy.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if self.index is None:
            if self.index_path.exists():
                self.load_index()
            else:
                self.build_index()

        if self.index is None:
            raise RuntimeError(
                "FAISS index is not initialized."
            )

        top_k = (
            top_k
            if top_k is not None
            else self.default_top_k
        )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = self.embedder.embed_query(
            query
        )

        query_embedding = query_embedding.reshape(
            1,
            -1,
        )

        search_k = min(
            max(top_k * 4, 20),
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding,
            search_k,
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):
            if index_id == -1:
                continue

            metadata = self.metadata[index_id]

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

            if len(results) >= top_k:
                break

        return results

    # ----------------------------------------------------------------
    # Print search results
    # ----------------------------------------------------------------

    @staticmethod
    def print_results(
        query: str,
        results: list[dict[str, Any]],
    ) -> None:

        print("\n" + "=" * 80)
        print("USER POLICY RETRIEVAL RESULTS")
        print("=" * 80)

        print(f"\nQuestion:")
        print(query)

        print(
            f"\nResults found: "
            f"{len(results)}"
        )

        for result in results:

            print("\n" + "-" * 80)

            print(
                f"Rank:       "
                f"{result['rank']}"
            )

            print(
                f"Score:      "
                f"{result['score']:.4f}"
            )

            print(
                f"Clause ID:  "
                f"{result['clause_id']}"
            )

            print(
                f"Policy ID:  "
                f"{result['policy_id']}"
            )

            print(
                f"Page:       "
                f"{result['page_start']}"
                f"-"
                f"{result['page_end']}"
            )

            if result["section"]:
                print(
                    f"Section:    "
                    f"{result['section']}"
                )

            if result["subsection"]:
                print(
                    f"Subsection: "
                    f"{result['subsection']}"
                )

            if result["marker"]:
                print(
                    f"Marker:     "
                    f"{result['marker']}"
                )

            print("\nClause:")
            print(result["clause_text"])

        print("\n" + "=" * 80)


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------

def main() -> None:

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "python part4_generation_evaluation/"
            "user_policy.py <path_to_pdf>"
        )
        sys.exit(1)

    pdf_path = sys.argv[1]

    policy = UserPolicy(
        pdf_path=pdf_path,
        enable_ocr=True,
        top_k=5,
    )

    # Always build the index for this standalone ingestion test.
    policy.build_index()

    print("\n" + "=" * 80)
    print("INTERACTIVE USER POLICY SEARCH")
    print("=" * 80)

    print(
        "\nAsk questions about the uploaded policy."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    while True:

        try:
            question = input(
                "\nQuestion: "
            ).strip()

        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if question.lower() in {
            "exit",
            "quit",
        }:
            print("Exiting.")
            break

        if not question:
            continue

        try:

            results = policy.search(
                query=question,
                top_k=5,
            )

            policy.print_results(
                query=question,
                results=results,
            )

        except Exception as error:

            print(
                f"\nRetrieval error: {error}"
            )


if __name__ == "__main__":
    main()