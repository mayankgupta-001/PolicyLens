# PolicyLens

## Part 1 — Policy Document Processing & Knowledge Preparation

Part 1 converts the selected health-insurance policy PDFs into a structured, clause-level dataset while preserving source page, section, subsection, and clause-marker information.

### Final Part 1 pipeline

```text
Health-insurance PDF
        ↓
Page-level text extraction
        ↓
OCR fallback when required
        ↓
Section / subsection detection
        ↓
Clause segmentation
        ↓
Page + source metadata preservation
        ↓
Light deterministic cleaning
        ↓
Automatic / pseudo annotation
        ↓
Structural validation
        ↓
765-clause frozen dataset
        ↓
Part 2: DeBERTa-v3-base classification
```

### Frozen corpus

The current experiment uses four policy documents:

| Policy | Clauses |
|---|---:|
| `POL001_HDFC_ERGO_EQUICOVE` | 325 |
| `POL002_LIC_JEEVAN_AROGYA` | 149 |
| `POL003_NIVA_BUPA_AROGYA_S` | 223 |
| `POL004_STAR_COMPREHENSIVE` | 68 |
| **Total** | **765** |

The dataset is automatically/pseudo-labeled. The labels are preliminary annotations used for model development; they are **not presented as human-verified insurance ground truth**.

### Final Part 1 output

`data/dataset_final_extraction/` contains:

- `all_clauses_annotation.csv` — consolidated 765-clause dataset.
- `POL001_HDFC_ERGO_EQUICOVE_clauses.json`
- `POL002_LIC_JEEVAN_AROGYA_clauses.json`
- `POL003_NIVA_BUPA_AROGYA_S_clauses.json`
- `POL004_STAR_COMPREHENSIVE_clauses.json`
- `policies.csv` — policy-level metadata and clause counts.

### Clause schema

```json
{
  "clause_id": "POL001_HDFC_ERGO_EQUICOVE_C0001",
  "policy_id": "POL001_HDFC_ERGO_EQUICOVE",
  "page_start": 1,
  "page_end": 1,
  "section": "1. Preamble",
  "subsection": "",
  "marker": "",
  "clause_text": "...",
  "category": ""
}
```

`category` is the preliminary taxonomy field used by the next stage. The five PolicyLens categories are:

- Coverage
- Exclusion
- Waiting Period
- Condition
- Claim Requirement

### Main Part 1 code

- `src/ingestion/pdf_extractor.py` — page-level PDF extraction with optional OCR fallback.
- `src/ingestion/text_cleaner.py` — extraction/text normalization utilities.
- `src/ingestion/clause_extractor.py` — deterministic, structure-aware clause segmentation. It does not perform AI classification.
- `src/ingestion/table_extractor.py` — table detection/extraction utility.
- `scripts/build_policy_dataset.py` — reproducible builder for the frozen four-policy corpus.
- `scripts/process_policy.py` — standalone PDF text/table extraction utility.
- `scripts/extract_clauses.py` — standalone clause extraction utility for development/debugging.
- `scripts/validate_clauses.py` — validation of the frozen Part 1 dataset.

### Build the frozen dataset

The final builder accepts only the four configured source PDFs and intentionally ignores unrelated PDFs such as PMSBY.

```bash
python scripts/build_policy_dataset.py
```

The builder checks the expected clause counts:

```text
POL001_HDFC_ERGO_EQUICOVE: 325
POL002_LIC_JEEVAN_AROGYA: 149
POL003_NIVA_BUPA_AROGYA_S: 223
POL004_STAR_COMPREHENSIVE: 68
```

### Validate the frozen dataset

```bash
python scripts/validate_clauses.py
```

The validator checks:

- total clause count
- per-policy counts
- duplicate clause IDs
- required fields
- empty clause text
- page ranges
- obvious page/header/footer furniture contamination
- unknown policy IDs

Clauses shorter than 20 characters are reported as a warning rather than automatically deleted because short policy statements can be legitimate.

### Design choice

Clause extraction is intentionally deterministic and inspectable. The extraction stage uses document structure, headings, clause markers, page boundaries, and continuation lines. AI classification is kept separate and belongs to Part 2.

OCR is available for difficult PDFs. Native/layout extraction remains the default because OCR can alter text and heading recognition.

## Repository scope

This repository is being developed in four logical parts:

1. **Part 1 — Policy Document Processing & Knowledge Preparation**
2. **Part 2 — Clause Intelligence & Classification (DeBERTa-v3-base)**
3. **Part 3 — Retrieval Intelligence & RAG (BGE-M3 / FAISS)**
4. **Part 4 — Answer Generation, Verification & Evaluation**

Part 1 is currently frozen at 765 clauses and has passed structural validation. The next stage uses this frozen dataset rather than regenerating the policy extraction.
