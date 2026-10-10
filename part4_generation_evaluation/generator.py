import os
import sys
import json

from google import genai
from google.genai import types


# ----------------------------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from part4_generation_evaluation.verifier import (
    PolicyAnswerVerifier
)


class PolicyGenerator:
    """
    Generates answers from retrieved policy clauses and independently
    verifies those answers against the same retrieved evidence.

    Pipeline:

        Retrieved Clauses
              ↓
        Gemini Generation
              ↓
        Gemini Verification
              ↓
        supported
             → final

        partially_supported
             → final

        unsupported
             → regenerate
             → verify again
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key=None,
        model=None
    ):
        self.model = model or self.DEFAULT_MODEL

        if api_key:
            self.client = genai.Client(
                api_key=api_key
            )
        else:
            self.client = genai.Client()

        self.verifier = PolicyAnswerVerifier(
            api_key=api_key,
            model=self.model
        )

    # ------------------------------------------------------------------
    # FORMAT CLAUSES
    # ------------------------------------------------------------------

    def _format_clauses(
        self,
        retrieved_clauses
    ):
        if not retrieved_clauses:
            return "NO RETRIEVED POLICY CLAUSES."

        formatted = []

        for i, clause in enumerate(
            retrieved_clauses,
            start=1
        ):
            clause_id = clause.get(
                "clause_id",
                "UNKNOWN"
            )

            policy_id = clause.get(
                "policy_id",
                "UNKNOWN"
            )

            page_start = clause.get(
                "page_start",
                ""
            )

            page_end = clause.get(
                "page_end",
                ""
            )

            section = clause.get(
                "section",
                ""
            )

            subsection = clause.get(
                "subsection",
                ""
            )

            marker = clause.get(
                "marker",
                ""
            )

            clause_text = clause.get(
                "clause_text",
                clause.get(
                    "text",
                    ""
                )
            )

            formatted.append(
                f"""
CLAUSE {i}

Clause ID:
{clause_id}

Policy ID:
{policy_id}

Page:
{page_start}-{page_end}

Section:
{section}

Subsection:
{subsection}

Marker:
{marker}

Text:
{clause_text}
"""
            )

        return "\n".join(formatted)

    # ------------------------------------------------------------------
    # GENERATION PROMPT
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question,
        retrieved_clauses
    ):
        evidence = self._format_clauses(
            retrieved_clauses
        )

        return f"""
You are PolicyLens, an AI assistant that explains health insurance
policy documents.

Answer the user's question using ONLY the retrieved policy clauses.

You MUST NOT use outside knowledge.

============================================================
USER QUESTION
============================================================

{question}

============================================================
RETRIEVED POLICY EVIDENCE
============================================================

{evidence}

============================================================
GROUNDING RULES
============================================================

1. Use only information present in the retrieved clauses.

2. Do not invent:
   - coverage
   - exclusions
   - waiting periods
   - percentages
   - monetary limits
   - co-pay
   - room-rent limits
   - conditions
   - claim requirements
   - policy rules

3. Every factual statement must be supported by retrieved evidence.

4. If the retrieved evidence does not contain enough information
   to answer the question, say so clearly.

5. Do not use general insurance knowledge to fill missing information.

6. Do not mix information from different policies.

7. Use the policy_id and clause_id from the retrieved evidence
   when providing sources.

============================================================
PARTIAL EVIDENCE
============================================================

The user may ask for several things in one question.

For example:

"Give the waiting period, cataract limit, co-pay and room-rent rule."

If the retrieved evidence contains only the waiting period and
hospitalization requirement:

- Answer those supported parts.
- Explicitly say that the other requested details are not
  established by the retrieved evidence.
- Do NOT invent the missing values.

In that situation:

evidence_status = "partially_supported"

============================================================
INSUFFICIENT EVIDENCE
============================================================

If the retrieved evidence contains no useful information for
answering the user's question:

evidence_status = "insufficient"

The answer should clearly state that the retrieved policy
information is insufficient.

============================================================
LANGUAGE RULES
============================================================

11. Explain the policy in simple language that an ordinary
    person can understand.

12. Prefer common words over insurance/legal terminology.

13. If an important insurance term is necessary, briefly explain
    it in simple language instead of only repeating the technical term.

14. Convert durations into familiar forms where possible.

    Example:
    "36 months (3 years)"

15. Do not remove important conditions, exclusions, limits,
    qualifications, or exceptions just to simplify the answer.

16. Do not change the meaning while simplifying.

17. Answer the actual question first, then briefly explain
    important conditions.

18. Write as if explaining the policy to someone who has never
    studied insurance terminology.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use exactly:

{{
    "answer": "clear answer to the user",
    "evidence_status": "supported",
    "sources": [
        {{
            "policy_id": "POLICY_ID",
            "clause_id": "CLAUSE_ID",
            "page": 1
        }}
    ]
}}

Allowed evidence_status values:

- supported
- partially_supported
- insufficient

If the evidence is insufficient, sources should be [].

If the evidence is partially supported, include sources for
the claims that ARE supported.
"""

    # ------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------

    def generate(
        self,
        question,
        retrieved_clauses
    ):
        prompt = self._build_prompt(
            question=question,
            retrieved_clauses=retrieved_clauses
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )

        raw_text = response.text.strip()

        try:
            result = json.loads(
                raw_text
            )
        except json.JSONDecodeError:
            return {
                "answer": (
                    "I could not generate a valid "
                    "evidence-grounded answer."
                ),
                "evidence_status": "insufficient",
                "sources": []
            }

        return self._validate_result(
            result
        )

    # ------------------------------------------------------------------
    # VALIDATE GENERATOR OUTPUT
    # ------------------------------------------------------------------

    def _validate_result(
        self,
        result
    ):
        if not isinstance(result, dict):
            return {
                "answer": (
                    "The retrieved policy information "
                    "is insufficient to answer this question."
                ),
                "evidence_status": "insufficient",
                "sources": []
            }

        answer = result.get(
            "answer",
            ""
        )

        evidence_status = result.get(
            "evidence_status",
            "insufficient"
        )

        sources = result.get(
            "sources",
            []
        )

        if not isinstance(answer, str):
            answer = str(answer)

        if evidence_status not in {
            "supported",
            "partially_supported",
            "insufficient"
        }:
            evidence_status = "insufficient"

        if not isinstance(sources, list):
            sources = []

        return {
            "answer": answer,
            "evidence_status": evidence_status,
            "sources": sources
        }

    # ------------------------------------------------------------------
    # CORRECTION PROMPT
    # ------------------------------------------------------------------

    def _build_correction_prompt(
        self,
        question,
        retrieved_clauses,
        previous_answer,
        verification
    ):
        evidence = self._format_clauses(
            retrieved_clauses
        )

        verification_json = json.dumps(
            verification,
            indent=2
        )

        return f"""
You are correcting an answer for PolicyLens.

The previous answer FAILED evidence verification.

You must generate a new answer that fixes the verifier's
identified problems.

============================================================
USER QUESTION
============================================================

{question}

============================================================
RETRIEVED POLICY EVIDENCE
============================================================

{evidence}

============================================================
PREVIOUS ANSWER
============================================================

{previous_answer}

============================================================
VERIFIER FEEDBACK
============================================================

{verification_json}

============================================================
CORRECTION RULES
============================================================

1. Correct every unsupported factual claim identified by the verifier.

2. Do not repeat an unsupported number, percentage, limit,
   waiting period, condition, or policy rule.

3. Do not invent information.

4. Use ONLY the retrieved policy clauses.

5. If a requested detail cannot be established from the retrieved
   clauses, explicitly say that the available evidence does not
   establish that detail.

6. Do not use outside knowledge.

7. Do not mix policies.

8. Preserve factual claims that ARE supported by the evidence.

9. If only some requested details are supported, use:

   evidence_status = "partially_supported"

10. If none of the requested information can be supported, use:

   evidence_status = "insufficient"

11. Use simple language.

12. Preserve important conditions and qualifications.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON:

{{
    "answer": "corrected answer",
    "evidence_status": "supported",
    "sources": [
        {{
            "policy_id": "POLICY_ID",
            "clause_id": "CLAUSE_ID",
            "page": 1
        }}
    ]
}}

Allowed evidence_status:

- supported
- partially_supported
- insufficient
"""

        return prompt

    # ------------------------------------------------------------------
    # REGENERATE
    # ------------------------------------------------------------------

    def _regenerate(
        self,
        question,
        retrieved_clauses,
        previous_result,
        verification
    ):
        previous_answer = previous_result.get(
            "answer",
            ""
        )

        prompt = self._build_correction_prompt(
            question=question,
            retrieved_clauses=retrieved_clauses,
            previous_answer=previous_answer,
            verification=verification
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )

        raw_text = response.text.strip()

        try:
            result = json.loads(
                raw_text
            )
        except json.JSONDecodeError:
            return {
                "answer": (
                    "I could not generate a valid "
                    "corrected answer from the retrieved "
                    "policy evidence."
                ),
                "evidence_status": "insufficient",
                "sources": []
            }

        return self._validate_result(
            result
        )

    # ------------------------------------------------------------------
    # VERIFIED GENERATION
    # ------------------------------------------------------------------

    def generate_verified(
        self,
        question,
        retrieved_clauses,
        max_attempts=3
    ):
        """
        Generate + verify an answer.

        Attempt behavior:

            supported
                -> return

            partially_supported
                -> return

            insufficient
                -> return

            unsupported
                -> regenerate

        Maximum number of attempts defaults to 3.
        """

        if not retrieved_clauses:
            return {
                "answer": (
                    "The retrieved policy information "
                    "is insufficient to answer this question."
                ),
                "evidence_status": "insufficient",
                "sources": [],
                "verification": {
                    "verified": False,
                    "overall_status": "insufficient",
                    "issues": [
                        "No relevant policy clauses were retrieved."
                    ],
                    "supported_claims": [],
                    "unsupported_claims": [],
                    "attempt": 0
                }
            }

        previous_result = None
        previous_verification = None

        for attempt in range(
            1,
            max_attempts + 1
        ):

            print()
            print("=" * 70)
            print(
                f"VERIFICATION ATTEMPT "
                f"{attempt}/{max_attempts}"
            )
            print("=" * 70)

            # ----------------------------------------------------------
            # FIRST ATTEMPT
            # ----------------------------------------------------------

            if attempt == 1:

                result = self.generate(
                    question=question,
                    retrieved_clauses=retrieved_clauses
                )

            # ----------------------------------------------------------
            # REGENERATION
            # ----------------------------------------------------------

            else:

                print(
                    "Regenerating answer using "
                    "verifier feedback..."
                )

                result = self._regenerate(
                    question=question,
                    retrieved_clauses=retrieved_clauses,
                    previous_result=previous_result,
                    verification=previous_verification
                )

            # ----------------------------------------------------------
            # INSUFFICIENT
            # ----------------------------------------------------------

            if result.get(
                "evidence_status"
            ) == "insufficient":

                result["verification"] = {
                    "verified": False,
                    "overall_status": "insufficient",
                    "issues": [
                        "The retrieved policy evidence "
                        "is insufficient to answer the question."
                    ],
                    "supported_claims": [],
                    "unsupported_claims": [],
                    "attempt": attempt
                }

                return result

            # ----------------------------------------------------------
            # VERIFY
            # ----------------------------------------------------------

            verification = self.verifier.verify(
                question=question,
                answer=result.get(
                    "answer",
                    ""
                ),
                retrieved_clauses=retrieved_clauses
            )

            verification["attempt"] = attempt

            result["verification"] = verification

            status = verification.get(
                "overall_status",
                "unsupported"
            )

            verified = verification.get(
                "verified",
                False
            )

            print(
                f"Verification: "
                f"{status.upper()}"
            )

            print(
                f"Verified: "
                f"{verified}"
            )

            # ----------------------------------------------------------
            # PRINT ISSUES
            # ----------------------------------------------------------

            issues = verification.get(
                "issues",
                []
            )

            if issues:

                print()
                print("Verifier issues:")

                for issue in issues:
                    print(
                        f"  - {issue}"
                    )

            unsupported_claims = verification.get(
                "unsupported_claims",
                []
            )

            if unsupported_claims:

                print()
                print("Unsupported claims:")

                for claim in unsupported_claims:

                    if isinstance(
                        claim,
                        dict
                    ):

                        print(
                            f"  - "
                            f"{claim.get('claim', '')}"
                        )

                        if claim.get(
                            "reason"
                        ):

                            print(
                                f"    Reason: "
                                f"{claim.get('reason')}"
                            )

                    else:

                        print(
                            f"  - {claim}"
                        )

            # ----------------------------------------------------------
            # FULLY SUPPORTED
            # ----------------------------------------------------------

            if (
                verified is True
                and status == "supported"
            ):

                print()
                print(
                    "✓ Answer passed verification."
                )

                return result

            # ----------------------------------------------------------
            # PARTIALLY SUPPORTED
            # ----------------------------------------------------------

            if (
                verified is True
                and status == "partially_supported"
            ):

                print()
                print(
                    "✓ Answer is partially supported."
                )

                print(
                    "  Returning it without regeneration "
                    "because it honestly reports missing evidence."
                )

                return result

            # ----------------------------------------------------------
            # UNSUPPORTED
            # ----------------------------------------------------------

            if status == "unsupported":

                print()
                print(
                    "✗ Answer failed verification."
                )

                if attempt < max_attempts:

                    print(
                        f"→ Starting attempt "
                        f"{attempt + 1}..."
                    )

                    previous_result = result
                    previous_verification = verification

                    continue

                # ------------------------------------------------------
                # ALL ATTEMPTS FAILED
                # ------------------------------------------------------

                print()
                print(
                    "✗ Maximum verification attempts reached."
                )

                return {
                    "answer": (
                        "I could not verify the generated "
                        "answer against the retrieved policy "
                        "clauses. The available policy evidence "
                        "is not sufficiently reliable to provide "
                        "a verified answer."
                    ),
                    "evidence_status": "insufficient",
                    "sources": [],
                    "verification": {
                        "verified": False,
                        "overall_status": "unsupported",
                        "issues": [
                            (
                                "The answer failed verification "
                                f"after {max_attempts} attempts."
                            )
                        ],
                        "supported_claims": verification.get(
                            "supported_claims",
                            []
                        ),
                        "unsupported_claims": verification.get(
                            "unsupported_claims",
                            []
                        ),
                        "attempt": attempt
                    }
                }

            # ----------------------------------------------------------
            # UNKNOWN STATUS
            # ----------------------------------------------------------

            print(
                "Unknown verifier status. "
                "Treating answer as failed."
            )

            previous_result = result
            previous_verification = verification

        # ----------------------------------------------------------------
        # SAFETY FALLBACK
        # ----------------------------------------------------------------

        return {
            "answer": (
                "The retrieved policy information "
                "could not be reliably verified."
            ),
            "evidence_status": "insufficient",
            "sources": [],
            "verification": {
                "verified": False,
                "overall_status": "unsupported",
                "issues": [
                    "Verification loop ended unexpectedly."
                ],
                "supported_claims": [],
                "unsupported_claims": [],
                "attempt": max_attempts
            }
        }


# ----------------------------------------------------------------------
# SIMPLE TEST
# ----------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("POLICY GENERATOR")
    print("=" * 70)

    generator = PolicyGenerator()

    print()
    print("PolicyGenerator loaded successfully.")