from __future__ import annotations

import json

from generator import PolicyGenerator
from pathlib import Path
import sys


# ---------------------------------------------------------
# Allow importing Part 3 from the project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part3_retrieval_rag.retriever import PolicyRetriever


class PolicyLensApp:
    """
    Connects the PolicyLens retrieval and generation layers.

    Flow:

        User question
              ↓
        FAISS retrieval
              ↓
        Retrieved clauses
              ↓
        Gemini generation
              ↓
        Grounded answer + sources
    """

    def __init__(self):
        print("=" * 70)
        print("INITIALIZING POLICYLENS")
        print("=" * 70)

        print("\nLoading retriever...")
        self.retriever = PolicyRetriever()

        print("\nLoading Gemini generator...")
        self.generator = PolicyGenerator()

        print("\nPolicyLens is ready.")

    # =========================================================
    # ASK QUESTION
    # =========================================================

    def ask(
        self,
        question: str,
        top_k: int = 5,
        policy_id: str | None = None,
    ) -> dict:

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # -----------------------------------------------------
        # Step 1: Retrieve relevant clauses
        # -----------------------------------------------------

        print("\nRetrieving relevant policy clauses...")

        retrieved_clauses = self.retriever.search(
            query=question,
            top_k=top_k,
            policy_id=policy_id,
        )

        if not retrieved_clauses:
            return {
                "answer": (
                    "No relevant policy clauses "
                    "were found."
                ),
                "evidence_status": "insufficient",
                "sources": [],
            }

        # -----------------------------------------------------
        # Step 2: Show retrieval information
        # -----------------------------------------------------

        print(
            f"Retrieved {len(retrieved_clauses)} "
            f"relevant clauses."
        )

        # -----------------------------------------------------
        # Step 3: Generate grounded answer
        # -----------------------------------------------------

        print(
            "Generating grounded answer..."
        )

        result = self.generator.generate(
            question=question,
            retrieved_clauses=retrieved_clauses,
        )

        # -----------------------------------------------------
        # Step 4: Return result
        # -----------------------------------------------------

        return result

    # =========================================================
    # PRINT ANSWER
    # =========================================================

    @staticmethod
    def print_answer(
        question: str,
        result: dict,
    ) -> None:

        print("\n")
        print("=" * 70)
        print("POLICYLENS ANSWER")
        print("=" * 70)

        print("\nQuestion:")
        print(question)

        print("\nAnswer:")
        print(result["answer"])

        print(
            "\nEvidence status:"
        )
        print(
            result["evidence_status"]
        )

        print("\nSources:")

        if not result["sources"]:
            print("No sources available.")

        else:
            for source in result["sources"]:

                print(
                    f"- Policy: "
                    f"{source.get('policy_id', 'N/A')}"
                )

                print(
                    f"  Clause: "
                    f"{source.get('clause_id', 'N/A')}"
                )

                print(
                    f"  Page: "
                    f"{source.get('page', 'N/A')}"
                )

        print("\n" + "=" * 70)

    # =========================================================
    # INTERACTIVE MODE
    # =========================================================

    def run(self):

        print("\n")
        print("=" * 70)
        print("POLICYLENS INSURANCE QA")
        print("=" * 70)

        print(
            "\nAsk questions about the indexed insurance policies."
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

                print(
                    "\n\nExiting."
                )

                break

            if question.lower() in {
                "exit",
                "quit",
            }:

                print(
                    "\nExiting."
                )

                break

            if not question:
                continue

            try:

                result = self.ask(
                    question=question,
                    top_k=5,
                )

                self.print_answer(
                    question,
                    result,
                )

            except Exception as error:

                print(
                    "\nPolicyLens error:"
                )

                print(error)


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    app = PolicyLensApp()

    app.run()