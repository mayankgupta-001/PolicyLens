# PolicyLens

## Part 1 — Policy Understanding & Structuring

The first module converts a health-insurance PDF into a structured, clause-level representation while preserving source page and section information.

### Pipeline

```text
PDF
 ↓
Visual text extraction / cleaning
 ↓
Section + subsection detection
 ↓
Clause marker detection and continuation merging
 ↓
Structured clause JSON
 ↓
Manual annotation
 ↓
DeBERTa-v3-base clause classification (next module)
```

### Current files

- `src/ingestion/pdf_extractor.py` — page-level PDF extraction with optional OCR fallback.
- `src/ingestion/table_extractor.py` — table detection.
- `src/ingestion/text_cleaner.py` — Unicode and extraction cleanup.
- `src/ingestion/clause_extractor.py` — deterministic clause segmentation. It does **not** classify clauses with AI.
- `scripts/process_policy.py` — PDF text + table extraction.
- `scripts/extract_clauses.py` — clause extraction from a PDF or extracted JSON.
- `scripts/validate_clauses.py` — basic structural validation.

### Clause schema

```json
{
  "clause_id": "POL001_C0001",
  "policy_id": "POL001",
  "page_start": 16,
  "page_end": 16,
  "section": "14. Waiting Period",
  "subsection": "Specific waiting period",
  "marker": "i.",
  "clause_text": "...",
  "source_lines": ["..."],
  "category": null
}
```

`category` is intentionally blank. The next stage will assign the PolicyLens taxonomy label using the annotated dataset and DeBERTa-v3-base.

### Run

```bash
pip install -r requirements.txt

# 1. Extract PDF text and tables
python scripts/process_policy.py data/raw_pdfs/policy.pdf

# 2. Segment the PDF into clauses
python scripts/extract_clauses.py data/raw_pdfs/policy.pdf --policy-id POL001

# 3. Validate the generated clauses
python scripts/validate_clauses.py
```

You can also extract clauses from an existing extraction JSON:

```bash
python scripts/extract_clauses.py data/extracted/policy_extracted.json
```

### Important design choice

Clause extraction is currently rule/structure based. The system uses headings, list markers, page boundaries, and continuation lines. This is deliberate: clause segmentation should be inspectable before supervised classification is trained.

OCR is available through `--ocr` / `--ocr-all` for difficult PDFs, but OCR can sometimes change heading recognition. The default native/layout extraction should therefore be inspected before using OCR output for annotation.

## Annotation stage

The clause extractor produces `data/clauses/policy_clauses_annotation.csv` with a blank `category` field. Use:

```bash
python scripts/annotate_clauses.py data/clauses/policy_clauses_annotation.csv
```

Labels:
- Coverage
- Exclusion
- Waiting Period
- Condition
- Claim Requirement

`data/clauses/policy_clauses_seed.csv` contains only high-confidence automatic suggestions. These are suggestions for annotation review, not ground truth.

See `annotation_guidelines.md` before labeling. The final classifier dataset should include clauses from multiple health-insurance policies and should preferably be split by policy/document to avoid train-test leakage.

## Part 1.5 — Multi-policy dataset preparation

The classifier must not be trained on only one policy. Put additional official health-insurance policy PDFs in `data/raw_pdfs/` and run:

```bash
python scripts/build_policy_dataset.py
```

This creates one clause JSON per policy plus:

- `data/dataset/all_clauses_annotation.csv`
- `data/dataset/policies.csv`

After manual annotation, split by **policy**, not by random clause, to reduce leakage:

```bash
python scripts/split_by_policy.py
```

A split requires at least 3 different policies. The final project should use more policies than this minimum.

## Suggested policy sources for dataset expansion

Use current policy-wording PDFs from official insurer websites. Examples include HDFC ERGO's health policy-wordings page and ICICI Lombard's Complete Health Insurance policy wording. These are source candidates for adding independent policies to the training corpus; the exact product/version should be recorded with the PDF.
