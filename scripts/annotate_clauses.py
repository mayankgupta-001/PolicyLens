#!/usr/bin/env python3
"""Interactive clause annotation tool for PolicyLens.

Usage:
    python scripts/annotate_clauses.py data/clauses/policy_clauses_annotation.csv

Keys:
    1 Coverage
    2 Exclusion
    3 Waiting Period
    4 Condition
    5 Claim Requirement
    s Skip / leave blank
    q Save and quit
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

LABELS = {
    "1": "Coverage",
    "2": "Exclusion",
    "3": "Waiting Period",
    "4": "Condition",
    "5": "Claim Requirement",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--start", type=int, default=0, help="Row index to start from")
    args = ap.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)
    if "category" not in df.columns:
        df["category"] = ""
    df["category"] = df["category"].fillna("")

    print("\nPolicyLens Clause Annotation Tool")
    print("1 Coverage | 2 Exclusion | 3 Waiting Period | 4 Condition | 5 Claim Requirement")
    print("s Skip | q Save & quit\n")

    for i in range(args.start, len(df)):
        row = df.iloc[i]
        print("=" * 90)
        print(f"[{i+1}/{len(df)}] {row['clause_id']} | pages {row['page_start']}-{row['page_end']}")
        print(f"Section: {row['section']}")
        if str(row.get('subsection', '')) not in ('', 'nan'):
            print(f"Subsection: {row['subsection']}")
        if str(row.get('marker', '')) not in ('', 'nan'):
            print(f"Marker: {row['marker']}")
        print(f"\n{row['clause_text']}\n")
        current = row["category"]
        if current:
            print(f"Current label: {current}")

        while True:
            choice = input("Label [1-5/s/q]: ").strip().lower()
            if choice in LABELS:
                df.at[i, "category"] = LABELS[choice]
                break
            if choice == "s":
                df.at[i, "category"] = ""
                break
            if choice == "q":
                df.to_csv(path, index=False)
                print(f"Saved: {path}")
                return
            print("Invalid choice.")

    df.to_csv(path, index=False)
    print(f"\nFinished. Saved: {path}")


if __name__ == "__main__":
    main()
