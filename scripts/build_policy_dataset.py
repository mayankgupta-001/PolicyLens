#!/usr/bin/env python3
"""Build the final four-policy Part 1 clause dataset.

This script deliberately selects only the four frozen PolicyLens policies,
keeps stable policy IDs, writes to data/dataset_final_extraction, and performs
light deterministic cleanup of repeated page furniture after clause extraction.
It does not assign PolicyLens categories.

Important: policy wording is not sentence-split or rewritten here. Clause text
is preserved as extracted, apart from whitespace normalization and removal of
obvious repeated page/header/footer furniture.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_extractor import extract_pdf
from src.ingestion.clause_extractor import extract_clauses


POLICIES = {
    "equicover": ("POL001_HDFC_ERGO_EQUICOVE", "HDFC_ERGO_EquiCover.pdf"),
    "lic": ("POL002_LIC_JEEVAN_AROGYA", "LIC_Jeevan_Arogya.pdf"),
    "niva": ("POL003_NIVA_BUPA_AROGYA_S", "Niva_Bupa_Arogya_Sanjeevani.pdf"),
    "star": ("POL004_STAR_COMPREHENSIVE", "Star_Comprehensive.pdf"),
}

EXPECTED_COUNTS = {
    "POL001_HDFC_ERGO_EQUICOVE": 325,
    "POL002_LIC_JEEVAN_AROGYA": 149,
    "POL003_NIVA_BUPA_AROGYA_S": 223,
    "POL004_STAR_COMPREHENSIVE": 68,
}

# Known legal/page furniture that must never become clause content.
FURNITURE_RE = [
    re.compile(r"^page\s+\d+(?:\s+of\s+\d+)?$", re.I),
    re.compile(r"^www\.[^\s]+$", re.I),
    re.compile(r"^https?://", re.I),
]


def clean_clause_text(text: str, repeated_lines: set[str]) -> str:
    """Clean whitespace/furniture without rewriting policy wording."""
    # Keep sentence boundaries and punctuation exactly as extracted.
    # Only collapse extraction whitespace and remove complete furniture lines.
    normalized_text = re.sub(r"\s+", " ", text.strip())
    if not normalized_text:
        return ""

    # A repeated header/footer may have been embedded inside a clause. Remove
    # it only when the complete normalized segment is a known repeated line.
    tokens = normalized_text.split(" ")
    kept_tokens: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            kept_tokens.extend(buffer)
            buffer.clear()

    # First handle the common case where the entire clause is furniture.
    low = normalized_text.casefold()
    if low in repeated_lines or any(p.search(normalized_text) for p in FURNITURE_RE):
        return ""

    # Do not attempt sentence-level legal rewriting. If furniture occurs as a
    # standalone newline in the original extraction, it has already been
    # separated before reaching this function in normal cases.
    # The whitespace normalization above is intentionally the only transformation.
    return normalized_text


def repeated_page_lines(extracted: dict) -> set[str]:
    """Find lines repeated on multiple pages, a strong header/footer signal."""
    counts = Counter()
    for page in extracted.get("pages", []):
        seen_on_page = set()
        for raw in (page.get("text") or "").splitlines():
            line = re.sub(r"\s+", " ", raw).strip().casefold()
            if len(line) >= 4:
                seen_on_page.add(line)
        for line in seen_on_page:
            counts[line] += 1

    page_count = max(1, len(extracted.get("pages", [])))
    return {
        line for line, count in counts.items()
        if count >= 2 and count / page_count >= 0.20 and len(line) <= 140
    }


def identify_policy(pdf: Path) -> tuple[str, str]:
    """Map the four frozen corpus PDFs using exact/strict filename rules."""
    name = pdf.name.casefold()

    if name == "equicover-health-cis-pw-108534555408.pdf":
        return POLICIES["equicover"]

    if name == "policy.pdf":
        return POLICIES["lic"]

    if name == "arogyasanjeevani-policydocument.pdf":
        return POLICIES["niva"]

    if name.startswith("brochure_star_comprehensive_insurance_policy"):
        return POLICIES["star"]

    raise ValueError(f"Unexpected PDF in corpus: {pdf.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw_pdfs")
    parser.add_argument("--output-dir", default="data/dataset_final_extraction")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()

    raw_dir = ROOT / args.raw_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw_dir.glob("*.pdf"))
    selected = []
    for pdf in pdfs:
        try:
            identify_policy(pdf)
            selected.append(pdf)
        except ValueError:
            continue

    if len(selected) != 4:
        names = ", ".join(p.name for p in selected)
        raise SystemExit(
            f"Expected exactly 4 target PDFs, found {len(selected)}: {names}"
        )

    all_rows = []
    policy_rows = []

    for pdf_path in selected:
        policy_id, canonical_name = identify_policy(pdf_path)
        extracted = extract_pdf(str(pdf_path), enable_ocr=args.ocr)
        clauses = extract_clauses(extracted, policy_id=policy_id)

        repeated = repeated_page_lines(extracted)

        cleaned = []
        for clause in clauses:
            c = dict(clause)
            c["clause_text"] = clean_clause_text(
                c.get("clause_text", ""), repeated
            )
            if not c["clause_text"]:
                continue
            cleaned.append(c)

        for i, c in enumerate(cleaned, start=1):
            c["clause_id"] = f"{policy_id}_C{i:04d}"

        policy_json = out_dir / f"{policy_id}_clauses.json"
        policy_json.write_text(
            json.dumps(
                {
                    "policy_id": policy_id,
                    "source_file": canonical_name,
                    "source_pdf": pdf_path.name,
                    "page_count": len(extracted["pages"]),
                    "clauses": cleaned,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        for c in cleaned:
            all_rows.append(
                {
                    "clause_id": c["clause_id"],
                    "policy_id": policy_id,
                    "source_file": canonical_name,
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "section": c.get("section", ""),
                    "subsection": c.get("subsection", ""),
                    "marker": c.get("marker", ""),
                    "clause_text": c["clause_text"],
                    "category": "",
                }
            )

        policy_rows.append(
            {
                "policy_id": policy_id,
                "source_file": canonical_name,
                "page_count": len(extracted["pages"]),
                "clause_count": len(cleaned),
            }
        )
        print(
            f"{pdf_path.name}: {len(cleaned)} clauses -> {policy_id} "
            f"(expected {EXPECTED_COUNTS[policy_id]})"
        )

    all_rows.sort(key=lambda r: (r["policy_id"], int(r["clause_id"].split("_C")[-1])))

    fields = list(all_rows[0])
    with (out_dir / "all_clauses_annotation.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    with (out_dir / "policies.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=policy_rows[0])
        writer.writeheader()
        writer.writerows(policy_rows)

    mismatches = [
        f"{r['policy_id']}: {r['clause_count']} != {EXPECTED_COUNTS[r['policy_id']]}"
        for r in policy_rows
        if r["clause_count"] != EXPECTED_COUNTS[r["policy_id"]]
    ]

    print(f"\nPolicies: {len(policy_rows)}")
    print(f"Total clauses: {len(all_rows)}")
    if mismatches:
        print("\nCOUNT MISMATCH:")
        for item in mismatches:
            print("  " + item)
        if not args.allow_count_mismatch:
            raise SystemExit(2)
    else:
        print("Expected counts: PASS")


if __name__ == "__main__":
    main()
