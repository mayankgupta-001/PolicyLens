import json
from google import genai
from google.genai import types


class PolicyAnswerVerifier:
    """
    Verifies a generated policy answer against the retrieved policy clauses.

    The verifier does NOT use outside knowledge.
    It checks whether the generated answer is actually supported
    by the retrieved evidence.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    # ------------------------------------------------------------------
    # FORMAT RETRIEVED CLAUSES
    # ------------------------------------------------------------------

    def _format_clauses(self, retrieved_clauses):
        if not retrieved_clauses:
            return "NO RETRIEVED POLICY CLAUSES."

        formatted = []

        for i, clause in enumerate(retrieved_clauses, start=1):
            clause_id = clause.get("clause_id", "UNKNOWN")
            policy_id = clause.get("policy_id", "UNKNOWN")

            page_start = clause.get("page_start", "")
            page_end = clause.get("page_end", "")

            section = clause.get("section", "")
            subsection = clause.get("subsection", "")

            text = clause.get(
                "clause_text",
                clause.get("text", "")
            )

            formatted.append(
                f"""
CLAUSE {i}
Clause ID: {clause_id}
Policy ID: {policy_id}
Page: {page_start}-{page_end}
Section: {section}
Subsection: {subsection}

Clause text:
{text}
"""
            )

        return "\n".join(formatted)

    # ------------------------------------------------------------------
    # VERIFICATION PROMPT
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question,
        answer,
        retrieved_clauses
    ):
        evidence = self._format_clauses(retrieved_clauses)

        return f"""
You are an independent evidence verifier for an insurance-policy
question-answering system.

Your job is NOT to answer the user's question.

Your job is to determine whether the proposed answer is supported
by the RETRIEVED POLICY CLAUSES.

You must use ONLY the retrieved clauses below.

Do NOT use:
- General insurance knowledge
- Internet knowledge
- Knowledge about the insurance company
- Knowledge from the original PDF that is NOT present in the
  retrieved clauses
- Assumptions

============================================================
USER QUESTION
============================================================

{question}

============================================================
PROPOSED ANSWER
============================================================

{answer}

============================================================
RETRIEVED POLICY CLAUSES
============================================================

{evidence}

============================================================
VERIFICATION RULES
============================================================

Check EVERY substantive factual claim in the proposed answer.

Pay special attention to:

1. Numbers
   - waiting periods
   - percentages
   - monetary limits
   - room-rent limits
   - co-pay
   - durations
   - dates

2. Conditions
   - declared conditions
   - accepted conditions
   - eligibility requirements
   - hospitalization requirements
   - exceptions

3. Coverage
   - whether something is covered
   - whether something is excluded
   - whether coverage is conditional

4. Policy identity
   - Do not allow clauses from another policy to support the answer.

5. Sources
   - The answer must be supported by the actual retrieved clauses.

============================================================
VERY IMPORTANT STATUS RULES
============================================================

Use exactly one of these statuses:

SUPPORTED:
Every substantive factual claim required by the user's question
is directly supported by the retrieved policy clauses.

PARTIALLY_SUPPORTED:
Some requested information is supported, but one or more requested
details cannot be established from the retrieved clauses.

For example:

User asks for:
- waiting period
- cataract limit
- co-pay
- room-rent rule

Retrieved evidence supports:
- waiting period
- hospitalization

but does NOT contain evidence for:
- cataract limit
- co-pay
- room-rent rule

Then the answer can be partially supported.

IMPORTANT:
If the answer HONESTLY says that a requested detail is not
available in the retrieved evidence, that statement itself is NOT
an unsupported claim.

Do NOT mark such an answer as unsupported merely because evidence
is missing.

UNSUPPORTED:
The proposed answer actually makes a factual claim that is:
- contradicted by the retrieved clauses,
- invented,
- unsupported by the retrieved clauses,
- or contains an incorrect number, limit, waiting period,
  percentage, condition, or policy rule.

============================================================
IMPORTANT DISTINCTION
============================================================

Missing evidence is NOT the same thing as an unsupported claim.

Example:

Answer:
"The retrieved clauses do not specify the co-pay percentage."

If the retrieved clauses genuinely do not contain the co-pay
percentage, this is an HONEST limitation.

That should contribute to:

overall_status = "partially_supported"

NOT:

overall_status = "unsupported"

However:

Answer:
"The co-pay is 10%."

If the retrieved clauses do not support 10%, this is an
UNSUPPORTED factual claim.

============================================================
VERIFIED FIELD
============================================================

Set:

verified = true

when the answer contains no unsupported factual claims.

This means both of these can have verified=true:

1. supported
2. partially_supported

Set:

verified = false

only when the answer contains an actual unsupported,
contradictory, or invented factual claim.

============================================================
RETURN FORMAT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "verified": true,
    "overall_status": "supported",
    "issues": [],
    "supported_claims": [
        {{
            "claim": "claim from the answer",
            "source_clause_ids": ["CLAUSE_ID"]
        }}
    ],
    "unsupported_claims": []
}}

Allowed overall_status values:

- supported
- partially_supported
- unsupported

If partially supported:

{{
    "verified": true,
    "overall_status": "partially_supported",
    "issues": [
        "The retrieved evidence does not establish the co-pay percentage."
    ],
    "supported_claims": [],
    "unsupported_claims": []
}}

If unsupported:

{{
    "verified": false,
    "overall_status": "unsupported",
    "issues": [
        "The answer states an unsupported cataract limit of Rs. 1,00,000."
    ],
    "supported_claims": [],
    "unsupported_claims": [
        {{
            "claim": "The cataract limit is Rs. 1,00,000.",
            "reason": "The retrieved policy clause does not support this amount."
        }}
    ]
}}

Do not provide markdown.
Do not provide explanations outside the JSON.
"""

    # ------------------------------------------------------------------
    # VERIFY
    # ------------------------------------------------------------------

    def verify(
        self,
        question,
        answer,
        retrieved_clauses
    ):
        prompt = self._build_prompt(
            question=question,
            answer=answer,
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
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            return {
                "verified": False,
                "overall_status": "unsupported",
                "issues": [
                    "Verifier returned invalid JSON."
                ],
                "supported_claims": [],
                "unsupported_claims": []
            }

        # --------------------------------------------------------------
        # VALIDATE VERIFIER OUTPUT
        # --------------------------------------------------------------

        allowed_statuses = {
            "supported",
            "partially_supported",
            "unsupported"
        }

        status = result.get(
            "overall_status",
            "unsupported"
        )

        if status not in allowed_statuses:
            status = "unsupported"

        verified = result.get(
            "verified",
            False
        )

        if not isinstance(verified, bool):
            verified = False

        issues = result.get(
            "issues",
            []
        )

        if not isinstance(issues, list):
            issues = [str(issues)]

        supported_claims = result.get(
            "supported_claims",
            []
        )

        if not isinstance(supported_claims, list):
            supported_claims = []

        unsupported_claims = result.get(
            "unsupported_claims",
            []
        )

        if not isinstance(unsupported_claims, list):
            unsupported_claims = []

        # --------------------------------------------------------------
        # SAFETY NORMALIZATION
        # --------------------------------------------------------------

        # If there are actual unsupported claims, the answer cannot
        # be considered verified.
        if unsupported_claims:
            verified = False
            status = "unsupported"

        # A partially supported answer is allowed to be verified.
        elif status == "partially_supported":
            verified = True

        # Fully supported answer must have no unsupported claims.
        elif status == "supported":
            verified = True

        else:
            verified = False
            status = "unsupported"

        return {
            "verified": verified,
            "overall_status": status,
            "issues": issues,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims
        }


if __name__ == "__main__":
    print("PolicyAnswerVerifier loaded successfully.")