from pathlib import Path
from typing import Any

import fitz


def extract_tables(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract credible table candidates; noisy detections are filtered out."""
    pdf_path = Path(pdf_path)
    document = fitz.open(pdf_path)
    tables: list[dict[str, Any]] = []

    for page_index, page in enumerate(document):
        if not hasattr(page, "find_tables"):
            continue
        try:
            result = page.find_tables()
        except Exception:
            continue

        kept = 0
        for table in result.tables:
            rows = table.extract()
            row_count = len(rows)
            col_count = max((len(r) for r in rows), default=0)
            non_empty = sum(1 for row in rows for cell in row if cell and str(cell).strip())

            if row_count < 2 or col_count < 2 or non_empty < 4:
                continue

            kept += 1
            tables.append({
                "table_id": f"P{page_index + 1}_T{kept}",
                "page": page_index + 1,
                "bbox": [round(v, 2) for v in table.bbox],
                "rows": rows,
            })

    return tables
