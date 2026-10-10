# PolicyLens Part 2 Dataset

Prepared for DeBERTa-v3-base clause classification.

## Source
The dataset is derived from the PolicyLens Part 1 annotation file `PolicyLens_corrected_clauses_v4.csv`.

## Selection
Only automatically annotated/pseudo-labeled rows with:
- `include_in_training = YES`
- `confidence = HIGH`
- `final_category` in the five target classes
- clause text length >= 20 characters
were included. Rows marked for manual review, definitions/skip, unlabeled rows, and text-quality review rows were not used.

## Labels
- Coverage
- Exclusion
- Waiting Period
- Condition
- Claim Requirement

## Split
70/15/15 approximately, random seed 42. Exact normalized duplicate clause texts are kept in the same split to reduce leakage. This is a pseudo-labeled development/training dataset, not a human-verified gold dataset.

The current automatically labeled training pool contains three policies (HDFC ERGO EquiCover, LIC Jeevan Arogya, and Niva Bupa Arogya Sanjeevani); Star Comprehensive has no HIGH-confidence rows marked YES for training in the source annotation file, so it is not silently added.
