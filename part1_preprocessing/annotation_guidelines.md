# PolicyLens Clause Annotation Guidelines

## Labels

### 1. Coverage
Use when the clause states a benefit, service, amount, or situation that the policy provides/pays/covers.

Examples:
- Hospital Cash Benefit is payable when the stated hospitalization requirements are met.
- Major Surgical Benefit is payable for an eligible surgery.
- Ambulance Benefit amount is payable when its conditions are met.

### 2. Exclusion
Use when the policy explicitly says a treatment, event, person, circumstance, or expense is not covered / not payable.

Examples:
- A benefit shall not be payable for a specified procedure.
- Treatment/event is excluded from the policy.

### 3. Waiting Period
Use when the clause defines a waiting period, its duration, start/end condition, or treatment that becomes covered only after a specified period.

Examples:
- General waiting period of X days.
- Specific waiting period for a disease/procedure.

### 4. Condition
Use when the clause imposes a rule, eligibility requirement, limit, threshold, duration, restriction, or prerequisite that is not primarily a waiting period or claim-document requirement.

Examples:
- Maximum number of hospitalization days.
- Minimum/maximum entry age.
- Benefit payable only if hospitalization occurs in India.
- Requirement that surgery be medically necessary.

### 5. Claim Requirement
Use when the clause specifies evidence, documents, proof, notification, forms, or other material required to submit/support a claim.

Examples:
- Proof of surgery must be provided.
- Claim documents must be submitted.

## Priority rule for ambiguous clauses

Use this order when more than one concept appears:

1. Waiting Period
2. Exclusion
3. Claim Requirement
4. Coverage
5. Condition

Use the label representing the clause's **main purpose**, not every word appearing in it.

## Important

Do not invent labels from general knowledge. Read the complete clause and surrounding section. If the clause boundary looks wrong, fix the extractor before using that row as training data.

The first LIC brochure is a **seed annotation set**, not the final training dataset. The final DeBERTa dataset should contain clauses from multiple health-insurance policies and should be split by policy/document where possible to reduce leakage.
