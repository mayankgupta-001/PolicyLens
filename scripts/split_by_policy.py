#!/usr/bin/env python3
"""Create train/validation/test CSVs using policy-level splitting."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="data/dataset/all_clauses_annotation.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--out", default="data/dataset/splits")
    args = p.parse_args()

    df = pd.read_csv(ROOT / args.csv)
    policies = sorted(df.policy_id.dropna().unique().tolist())
    if len(policies) < 3:
        raise SystemExit(
            f"Need at least 3 policies for a policy-level train/val/test split; found {len(policies)}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(policies)

    n = len(policies)
    n_train = max(1, round(n * args.train))
    n_val = max(1, round(n * args.val))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_p = set(policies[:n_train])
    val_p = set(policies[n_train:n_train+n_val])
    test_p = set(policies[n_train+n_val:])

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    df[df.policy_id.isin(train_p)].to_csv(out / "train.csv", index=False)
    df[df.policy_id.isin(val_p)].to_csv(out / "validation.csv", index=False)
    df[df.policy_id.isin(test_p)].to_csv(out / "test.csv", index=False)

    print("Train policies:", sorted(train_p))
    print("Validation policies:", sorted(val_p))
    print("Test policies:", sorted(test_p))


if __name__ == "__main__":
    main()
