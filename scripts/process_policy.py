import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_extractor import extract_pdf
from src.ingestion.table_extractor import extract_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="PolicyLens Part 1 PDF extraction")
    parser.add_argument("pdf", help="Path to an insurance policy PDF")
    parser.add_argument("--out", default="data/extracted", help="Output directory")
    parser.add_argument("--ocr", action="store_true", help="Use OCR on pages with poor native extraction")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = extract_pdf(pdf_path, enable_ocr=args.ocr)
    tables = extract_tables(pdf_path)
    extracted["tables"] = tables

    output_path = out_dir / f"{pdf_path.stem}_extracted.json"
    output_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Pages: {extracted['page_count']}")
    print(f"Characters: {extracted['total_characters']}")
    print(f"Tables detected: {len(tables)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
