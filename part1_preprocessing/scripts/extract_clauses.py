import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from part1_preprocessing.src.ingestion.clause_extractor import extract_clauses, extract_clauses_from_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="PolicyLens clause segmentation")
    parser.add_argument("input", help="Extracted JSON file or source PDF")
    parser.add_argument("--out", default="data/clauses", help="Output directory")
    parser.add_argument("--policy-id", default="POL001")
    parser.add_argument("--min-chars", type=int, default=35)
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="If input is a PDF, use OCR fallback on low-quality pages",
    )
    parser.add_argument(
        "--ocr-all",
        action="store_true",
        help="If input is a PDF, OCR every page for better visual reading order",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".pdf":
        clauses = extract_clauses_from_pdf(
            str(input_path),
            policy_id=args.policy_id,
            min_clause_chars=args.min_chars,
            use_ocr=(args.ocr or args.ocr_all),
        )
        source_stem = input_path.stem
    else:
        extracted = json.loads(input_path.read_text(encoding="utf-8"))
        source_stem = input_path.stem.replace("_extracted", "")
        clauses = extract_clauses(
            extracted,
            policy_id=args.policy_id,
            min_clause_chars=args.min_chars,
        )

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{source_stem}_clauses.json"
    output_path.write_text(
        json.dumps(clauses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Clauses extracted: {len(clauses)}")
    print(f"Saved: {output_path}")
    print("\nFirst 10 clauses:")
    for clause in clauses[:10]:
        print(
            f"{clause['clause_id']} | p.{clause['page_start']}-{clause['page_end']} | "
            f"{clause['section']} | {clause['subsection']} | {clause['marker']} | {clause['clause_text'][:180]}"
        )


if __name__ == "__main__":
    main()
