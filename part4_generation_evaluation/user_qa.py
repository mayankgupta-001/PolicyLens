from __future__ import annotations

import sys
from pathlib import Path


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# PolicyLens modules
# ---------------------------------------------------------

from part4_generation_evaluation.generator import (
    PolicyGenerator,
)

from part4_generation_evaluation.user_policy import (
    UserPolicy,
)


class UserPolicyQA:
    """
    Complete QA pipeline for a user-provided insurance PDF.

    Flow:

    User PDF
        ↓
    Part 1 extraction
        ↓
    Structure-aware clauses
        ↓
    BGE-M3 embeddings
        ↓
    FAISS
        ↓
    Top-K retrieval
        ↓
    Gemini answer generation
        ↓
    Independent answer verification
        ↓
    Regeneration if verification fails
        ↓
    Final verified answer
    """

    def __init__(
        self,
        pdf_path: str | Path,
    ):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found:\n{self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "The provided file must be a PDF."
            )

        print("=" * 80)
        print("POLICYLENS USER POLICY QA")
        print("=" * 80)

        print(
            f"\nPolicy PDF:\n"
            f"{self.pdf_path}"
        )

        # -------------------------------------------------
        # User policy
        # -------------------------------------------------

        self.policy = UserPolicy(
            self.pdf_path
        )

        print("\nBuilding user policy index...")

        self.policy.build_index()

        # -------------------------------------------------
        # Generator
        # -------------------------------------------------

        print("\nLoading Gemini generator...")

        self.generator = PolicyGenerator()

        print("\nUser Policy QA ready.")

    # =====================================================
    # ASK QUESTION
    # =====================================================

    def ask(
        self,
        question: str,
        top_k: int = 8,
    ) -> dict:
        """
        Ask a question against the uploaded policy.

        top_k=8 provides more evidence to the generator
        and verifier than the earlier top_k=5 setup.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # -------------------------------------------------
        # Retrieval
        # -------------------------------------------------

        print(
            f"\nRetrieving top {top_k} policy clauses..."
        )

        retrieved_clauses = self.policy.search(
            query=question,
            top_k=top_k,
        )

        print(
            f"Retrieved clauses: "
            f"{len(retrieved_clauses)}"
        )

        if not retrieved_clauses:

            return {
                "answer": (
                    "The retrieved policy information "
                    "is insufficient to answer this question."
                ),
                "evidence_status": "insufficient",
                "sources": [],
                "retrieved_clauses": [],
                "verification": {
                    "verified": False,
                    "overall_status": "unsupported",
                    "issues": [
                        "No relevant policy clauses were retrieved."
                    ],
                    "supported_claims": [],
                    "unsupported_claims": [],
                    "attempt": 0,
                },
            }

        # -------------------------------------------------
        # Verified generation
        # -------------------------------------------------

        print(
            "\nGenerating and verifying answer..."
        )

        result = self.generator.generate_verified(
            question=question,
            retrieved_clauses=retrieved_clauses,
            max_attempts=3,
        )

        # Keep retrieved clauses available for debugging
        # and evaluation.

        result["retrieved_clauses"] = (
            retrieved_clauses
        )

        return result

    # =====================================================
    # PRINT ANSWER
    # =====================================================

    @staticmethod
    def print_answer(
        question: str,
        result: dict,
    ) -> None:

        print("\n")
        print("=" * 80)
        print("POLICYLENS ANSWER")
        print("=" * 80)

        print("\nQuestion:")
        print(question)

        print("\nAnswer:")
        print(result.get("answer", ""))

        print("\nEvidence status:")
        print(
            result.get(
                "evidence_status",
                "unknown",
            )
        )

        # -------------------------------------------------
        # Verification
        # -------------------------------------------------

        verification = result.get(
            "verification"
        )

        if verification:

            print("\nVerification:")
            print(
                f"Verified: "
                f"{verification.get('verified')}"
            )

            print(
                f"Status: "
                f"{verification.get('overall_status')}"
            )

            print(
                f"Attempt: "
                f"{verification.get('attempt', 0)}"
            )

            issues = verification.get(
                "issues",
                [],
            )

            if issues:

                print(
                    "\nVerification issues:"
                )

                for issue in issues:
                    print(
                        f"- {issue}"
                    )

            unsupported_claims = (
                verification.get(
                    "unsupported_claims",
                    [],
                )
            )

            if unsupported_claims:

                print(
                    "\nUnsupported claims:"
                )

                for claim in unsupported_claims:
                    print(
                        f"- {claim}"
                    )

        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        sources = result.get(
            "sources",
            [],
        )

        print("\nSources:")

        if not sources:
            print("No verified sources.")

        else:

            for source in sources:

                print(
                    f"- Policy: "
                    f"{source.get('policy_id')}"
                )

                print(
                    f"  Clause: "
                    f"{source.get('clause_id')}"
                )

                print(
                    f"  Page: "
                    f"{source.get('page')}"
                )

        # -------------------------------------------------
        # Retrieved clauses
        # -------------------------------------------------

        retrieved = result.get(
            "retrieved_clauses",
            [],
        )

        print(
            f"\nRetrieved clauses: "
            f"{len(retrieved)}"
        )

        print(
            "\n" + "=" * 80
        )

    # =====================================================
    # INTERACTIVE MODE
    # =====================================================

    def run(self) -> None:

        print("\n")
        print("=" * 80)
        print("INTERACTIVE POLICY QA")
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

                print(
                    "\n\nExiting."
                )

                break

            except EOFError:

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
                    top_k=8,
                )

                self.print_answer(
                    question=question,
                    result=result,
                )

            except Exception as error:

                print(
                    "\nERROR:"
                )

                print(error)


# =========================================================
# CLI
# =========================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python "
            "part4_generation_evaluation/"
            "user_qa.py "
            "\"path/to/policy.pdf\""
        )

        sys.exit(1)

    pdf_path = sys.argv[1]

    try:

        qa = UserPolicyQA(
            pdf_path
        )

        qa.run()

    except Exception as error:

        print(
            "\nFailed to start PolicyLens:"
        )

        print(error)

        sys.exit(1)


if __name__ == "__main__":
    main()