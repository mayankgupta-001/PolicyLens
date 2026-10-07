#!/usr/bin/env python3
"""Build a multi-policy clause dataset from PDFs in data/raw_pdfs.

Each PDF is processed independently, then clauses are merged into a single
annotation-ready CSV. Policy-level IDs are used so train/validation/test
splits can later be performed by policy rather than by individual clause.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_extractor import extract_pdf
from src.ingestion.clause_extractor import extract_clauses


def slug_policy_id(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").upper()
    stem = stem[:18] or f"POLICY_{index:03d}"
    return f"POL{index:03d}_{stem}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw_pdfs")
    parser.add_argument("--output-dir", default="data/dataset")
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()

    raw_dir = ROOT / args.raw_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {raw_dir}")

    all_rows = []
    policy_rows = []

    for idx, pdf_path in enumerate(pdfs, start=1):
        policy_id = slug_policy_id(pdf_path, idx)
        extracted = extract_pdf(str(pdf_path), enable_ocr=args.ocr)
        clauses = extract_clauses(extracted, policy_id=policy_id)

        policy_json = out_dir / f"{policy_id}_clauses.json"
        policy_json.write_text(
            json.dumps({
                "policy_id": policy_id,
                "source_file": pdf_path.name,
                "page_count": len(extracted["pages"]),
                "clauses": clauses,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for c in clauses:
            all_rows.append({
                "clause_id": c["clause_id"],
                "policy_id": policy_id,
                "source_file": pdf_path.name,
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "section": c.get("section", ""),
                "subsection": c.get("subsection", ""),
                "marker": c.get("marker", ""),
                "clause_text": c["clause_text"],
                "category": "",
            })

        policy_rows.append({
            "policy_id": policy_id,
            "source_file": pdf_path.name,
            "page_count": len(extracted["pages"]),
            "clause_count": len(clauses),
        })
        print(f"{pdf_path.name}: {len(clauses)} clauses -> {policy_id}")

    fieldnames = list(all_rows[0].keys())
    with (out_dir / "all_clauses_annotation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with (out_dir / "policies.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=policy_rows[0].keys())
        writer.writeheader()
        writer.writerows(policy_rows)

    print(f"\nTotal policies: {len(policy_rows)}")
    print(f"Total clauses:  {len(all_rows)}")
    print(f"Annotation CSV: {out_dir / 'all_clauses_annotation.csv'}")


if __name__ == "__main__":
    main()
