# PolicyLens Part 2 — DeBERTa Training

This package contains the prepared dataset and the DeBERTa-v3-base training script.

## Run

From this folder:

```bash
python train_deberta.py
```

The script uses `microsoft/deberta-v3-base`, checks the GPU forward pass before training,
uses conservative settings for the RTX 3050 6 GB, evaluates validation/test performance,
prints a confusion matrix and per-class metrics, and saves the trained model under
`model/final/`.

The labels are automatically annotated/pseudo-labeled insurance clauses, not human-verified
gold labels. Therefore, metrics should be described as development/pseudo-label results,
not definitive research results.
