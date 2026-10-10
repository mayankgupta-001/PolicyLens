"""
PolicyLens Part 2 — DeBERTa-v3-base clause classifier.

Input:
    dataset/train.csv
    dataset/validation.csv
    dataset/test.csv

Output:
    model/best/
    model/final/

Designed for the current environment used in PolicyLens:
    PyTorch 2.11 + CUDA 12.8
    Transformers 5.19.x
    NVIDIA RTX 3050 6 GB

Important:
- Labels are automatically annotated / pseudo-labeled, not human-verified.
- This script does not hide NaN/Inf losses. It stops if numerical instability appears.
- FP16/BF16 and TF32 are intentionally disabled for stability.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
BEST_DIR = MODEL_DIR / "best"
FINAL_DIR = MODEL_DIR / "final"

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
NUM_EPOCHS = 4
LEARNING_RATE = 5e-6
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10
SEED = 42

LABELS = [
    "Coverage",
    "Exclusion",
    "Waiting Period",
    "Condition",
    "Claim Requirement",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

REQUIRED_COLUMNS = {
    "clause_id",
    "policy_id",
    "page_start",
    "page_end",
    "section",
    "subsection",
    "clause_text",
    "label",
}


def check_environment() -> None:
    print("=" * 72)
    print("POLICYLENS PART 2 — DeBERTa-v3-base TRAINING")
    print("=" * 72)
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA available:{torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:           {torch.cuda.get_device_name(0)}")
        print(f"GPU memory:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA is unavailable. Training will use CPU.")


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df["clause_text"] = df["clause_text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].fillna("").astype(str).str.strip()

    if df["clause_text"].eq("").any():
        raise ValueError(f"{path.name} contains empty clause_text values.")

    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"{path.name} contains unknown labels: {unknown}")

    if df["clause_id"].duplicated().any():
        dupes = df.loc[df["clause_id"].duplicated(), "clause_id"].tolist()
        raise ValueError(f"{path.name} contains duplicate clause IDs: {dupes[:10]}")

    return df


def check_no_text_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    def normalize(series: pd.Series) -> set[str]:
        return set(
            series.str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    sets = {
        "train": normalize(train["clause_text"]),
        "validation": normalize(val["clause_text"]),
        "test": normalize(test["clause_text"]),
    }

    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(
                f"Text leakage detected between {a} and {b}: {len(overlap)} duplicated texts."
            )


def print_dataset_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name}: {len(df)} rows")
    print(df["label"].value_counts().reindex(LABELS, fill_value=0).to_string())


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def make_training_args() -> TrainingArguments:
    # Transformers has changed argument names across releases.
    # Build only arguments supported by the installed version.
    params = inspect.signature(TrainingArguments.__init__).parameters

    kwargs = {
        "output_dir": str(BEST_DIR),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "max_grad_norm": 1.0,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": False,
        "bf16": False,
        "dataloader_num_workers": 0,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
    }

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"

    # Explicitly disable TF32 because the earlier PolicyLens run became unstable.
    if "tf32" in params:
        kwargs["tf32"] = False

    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**kwargs)


class WeightedLossTrainer(Trainer):
    """Trainer using inverse-frequency class-weighted cross entropy."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(device=logits.device, dtype=logits.dtype)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main() -> None:
    set_seed(SEED)
    check_environment()

    train_df = load_split(DATA_DIR / "train.csv")
    val_df = load_split(DATA_DIR / "validation.csv")
    test_df = load_split(DATA_DIR / "test.csv")

    check_no_text_leakage(train_df, val_df, test_df)

    print_dataset_summary("TRAIN", train_df)
    print_dataset_summary("VALIDATION", val_df)
    print_dataset_summary("TEST", test_df)

    # Balanced class weights: inverse frequency, normalized to mean 1.
    counts = train_df["label"].value_counts().reindex(LABELS, fill_value=0)
    weights = len(train_df) / (len(LABELS) * counts)
    class_weights = torch.tensor(weights.values, dtype=torch.float32)
    print("\nClass weights:")
    for label, weight in zip(LABELS, class_weights.tolist()):
        print(f"  {label}: {weight:.4f}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["clause_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    def to_hf_dataset(df: pd.DataFrame) -> Dataset:
        work = df[["clause_text", "label"]].copy()
        work["labels"] = work["label"].map(LABEL2ID)
        work = work.drop(columns=["label"])
        ds = Dataset.from_pandas(work, preserve_index=False)
        return ds.map(tokenize, batched=True, remove_columns=["clause_text"])

    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)
    test_ds = to_hf_dataset(test_df)

    print("\nLoading DeBERTa-v3-base...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Sanity-check model parameters and one GPU forward pass before training.
    all_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    print(f"Initial model parameters finite: {all_finite}")
    if not all_finite:
        raise RuntimeError("Initial model contains NaN/Inf parameters.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sample = tokenizer(
        train_df.iloc[0]["clause_text"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    sample = {k: v.to(device) for k, v in sample.items()}

    with torch.no_grad():
        logits = model(**sample).logits
        loss = torch.nn.functional.cross_entropy(
            logits,
            torch.tensor([LABEL2ID[train_df.iloc[0]["label"]]], device=device),
        )

    print(
        f"Pre-training sanity check: "
        f"logits_finite={torch.isfinite(logits).all().item()}, "
        f"loss_finite={torch.isfinite(loss).item()}, "
        f"loss={loss.item():.6f}"
    )

    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("Pre-training GPU sanity check failed.")

    BEST_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = make_training_args()

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedLossTrainer(
        **trainer_kwargs,
        class_weights=class_weights,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print("\nValidation evaluation...")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    print(json.dumps(val_metrics, indent=2, default=str))

    print("\nTest evaluation...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2, default=str))

    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
    )

    print("\nTEST CONFUSION MATRIX")
    print("Rows = actual, columns = predicted")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )

    per_class = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        index=LABELS,
    )

    print("\nPER-CLASS TEST METRICS")
    print(per_class.to_string())

    # Save the best model selected using validation Macro-F1.
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    results = {
        "training_version": "v2_class_weighted_loss",
        "model": MODEL_NAME,
        "labels": LABELS,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class.to_dict(orient="index"),
        "note": "Metrics are based on automatically annotated/pseudo-labeled data and are not human-verified gold-label results.",
    }

    with open(FINAL_DIR / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nModel saved to: {FINAL_DIR}")
    print("\nPART 2 TRAINING COMPLETE")
    print("NOTE: Treat the reported metrics as pseudo-label evaluation results.")


if __name__ == "__main__":
    main()
