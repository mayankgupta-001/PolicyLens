from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz

from .text_cleaner import clean_text, extraction_quality
import re

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def ocr_page_bytes(page_bytes: bytes, width: int, height: int) -> str:
    image = Image.frombytes("RGB", [width, height], page_bytes)
    return clean_text(pytesseract.image_to_string(image, config="--psm 6", lang="eng"))


def render_for_ocr(page: fitz.Page, dpi: int = 100):
    scale = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return pix.samples, pix.width, pix.height


def extract_pdf(pdf_path: str | Path, enable_ocr: bool = False, max_workers: int = 1, force_ocr: bool = False) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    document = fitz.open(pdf_path)
    pages = []
    ocr_jobs = []

    for page_index, page in enumerate(document):
        native_text = clean_text(page.get_text("text"))
        quality = extraction_quality(native_text)
        pages.append({
            "page": page_index + 1,
            "native_text": native_text,
            "native_quality": round(quality, 4),
            "text": native_text,
            "used_ocr": False,
            "char_count": len(native_text),
        })
        bad_chars = len(re.findall(r"[\x80-\x9f\ufffd]", native_text))
        if enable_ocr and OCR_AVAILABLE and (force_ocr or quality < 0.94 or bad_chars >= 3):
            samples, width, height = render_for_ocr(page)
            ocr_jobs.append((page_index, samples, width, height))

    # OCR only the pages where native extraction is likely corrupted.
    if ocr_jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(ocr_page_bytes, samples, width, height): page_index
                for page_index, samples, width, height in ocr_jobs
            }
            for future in as_completed(futures):
                page_index = futures[future]
                try:
                    ocr_text = future.result()
                except Exception:
                    continue
                native = pages[page_index]["native_text"]
                if len(ocr_text) >= max(100, int(len(native) * 0.35)):
                    pages[page_index]["text"] = ocr_text
                    pages[page_index]["used_ocr"] = True
                    pages[page_index]["char_count"] = len(ocr_text)

    return {
        "source_file": pdf_path.name,
        "page_count": len(document),
        "total_characters": sum(p["char_count"] for p in pages),
        "ocr_available": OCR_AVAILABLE,
        "pages": pages,
    }
