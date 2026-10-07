import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/clauses/policy_clauses.json")
    clauses = json.loads(path.read_text(encoding="utf-8"))

    required = {
        "clause_id", "policy_id", "page_start", "page_end",
        "section", "subsection", "marker", "clause_text", "category"
    }
    missing = [c["clause_id"] for c in clauses if not required.issubset(c)]
    duplicate_ids = [k for k, v in Counter(c["clause_id"] for c in clauses).items() if v > 1]
    bad_pages = [c["clause_id"] for c in clauses if c["page_start"] > c["page_end"]]

    sections = sorted(
        {c["section"] for c in clauses if c["section"]},
        key=lambda s: int(s.split(".")[0]) if s[0].isdigit() else 999,
    )

    print(f"Clauses: {len(clauses)}")
    print(f"Sections: {len(sections)}")
    print(f"Page range: {min(c['page_start'] for c in clauses)}-{max(c['page_end'] for c in clauses)}")
    print(f"Missing fields: {len(missing)}")
    print(f"Duplicate IDs: {len(duplicate_ids)}")
    print(f"Invalid page ranges: {len(bad_pages)}")
    print("\nSections:")
    for section in sections:
        print(f"  - {section}")

    if missing or duplicate_ids or bad_pages:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
