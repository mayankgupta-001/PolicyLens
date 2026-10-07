# Part 1 — Policy Document Processing & Knowledge Preparation

This module converts the four health-insurance policy PDFs into a structured, clause-level dataset while preserving page, section, subsection, marker, and source metadata.

## Pipeline

```text
PDF → text extraction → OCR fallback → section detection → clause segmentation → metadata preservation → cleaning → automatic/pseudo annotation → validation
```

## Final corpus

- POL001_HDFC_ERGO_EQUICOVE: 325 clauses
- POL002_LIC_JEEVAN_AROGYA: 149 clauses
- POL003_NIVA_BUPA_AROGYA_S: 223 clauses
- POL004_STAR_COMPREHENSIVE: 68 clauses
- Total: 765 clauses

The labels are automatic/pseudo annotations and are not presented as human-verified insurance ground truth.

## Main output

`data/dataset_final_extraction/` contains the four policy JSON files, `all_clauses_annotation.csv`, and `policies.csv`.

## Validation

Run from the repository root:

```powershell
python part1_preprocessing/scripts/validate_clauses.py
```

The current validated dataset has 765 clauses, zero duplicate IDs, zero missing fields, zero empty text, zero bad page ranges, zero detected furniture contamination, and zero unknown policies. Clauses shorter than 20 characters are warnings only.
