#!/usr/bin/env python3
"""Validate the frozen four-policy Part 1 extraction output."""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "POL001_HDFC_ERGO_EQUICOVE": 325,
    "POL002_LIC_JEEVAN_AROGYA": 149,
    "POL003_NIVA_BUPA_AROGYA_S": 223,
    "POL004_STAR_COMPREHENSIVE": 68,
}

FURNITURE = [
    re.compile(r"^page\s+\d+(?:\s+of\s+\d+)?$", re.I),
    re.compile(r"^www\.", re.I),
]


def main() -> None:
    out_dir = ROOT / (
        sys.argv[1] if len(sys.argv) > 1 else "data/dataset_final_extraction"
    )
    csv_path = out_dir / "all_clauses_annotation.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing: {csv_path}")

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {
        "clause_id", "policy_id", "source_file", "page_start", "page_end",
        "section", "subsection", "marker", "clause_text", "category"
    }
    missing_fields = [
        r.get("clause_id", "<missing>")
        for r in rows
        if not required.issubset(r)
    ]

    ids = Counter(r["clause_id"] for r in rows)
    duplicate_ids = [k for k, v in ids.items() if v > 1]

    policy_counts = Counter(r["policy_id"] for r in rows)
    unknown_policies = sorted(set(policy_counts) - set(EXPECTED))

    # Short clauses are reported as a warning, not a validation failure.
    # A short policy clause can be legitimate (e.g. a limit, age, or
    # applicability statement), so we must not delete it automatically.
    short_text = [
        r["clause_id"] for r in rows
        if len(r["clause_text"].strip()) < 20
    ]
    empty_text = [
        r["clause_id"] for r in rows
        if not r["clause_text"].strip()
    ]

    bad_pages = []
    for r in rows:
        try:
            start, end = int(r["page_start"]), int(r["page_end"])
            if start < 1 or end < start:
                bad_pages.append(r["clause_id"])
        except ValueError:
            bad_pages.append(r["clause_id"])

    furniture_hits = []
    for r in rows:
        text = r["clause_text"].strip()
        if any(p.search(text) for p in FURNITURE):
            furniture_hits.append(r["clause_id"])

    print("=== PolicyLens Part 1 Validation ===")
    print(f"Total clauses: {len(rows)}")
    print(f"Expected total: {sum(EXPECTED.values())}")
    print("\nPolicy counts:")
    for policy, expected in EXPECTED.items():
        actual = policy_counts.get(policy, 0)
        status = "PASS" if actual == expected else "FAIL"
        print(f"  {policy}: {actual} / {expected} [{status}]")

    print(f"\nDuplicate IDs: {len(duplicate_ids)}")
    print(f"Missing fields: {len(missing_fields)}")
    print(f"Empty text: {len(empty_text)}")
    print(f"Short text (<20 chars): {len(short_text)} [WARNING ONLY]")
    print(f"Bad page ranges: {len(bad_pages)}")
    print(f"Possible furniture contamination: {len(furniture_hits)}")
    print(f"Unknown policies: {len(unknown_policies)}")

    failures = (
        duplicate_ids
        or missing_fields
        or empty_text
        or bad_pages
        or furniture_hits
        or unknown_policies
        or len(rows) != sum(EXPECTED.values())
        or any(policy_counts.get(p, 0) != n for p, n in EXPECTED.items())
    )

    if failures:
        print("\nVALIDATION: FAIL")
        raise SystemExit(1)

    print("\nVALIDATION: PASS")


if __name__ == "__main__":
    main()
