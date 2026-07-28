"""
Multimodal PDF extraction: text layout, tables, embedded images, and optional OCR
for low-text or scanned pages. Designed to feed the same chunking + embedding path as plain text.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import fitz  # PyMuPDF

from app.config import Settings

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[misc, assignment]

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[misc, assignment]


@dataclass
class ExtractionStats:
    page_count: int = 0
    tables_extracted: int = 0
    images_found: int = 0
    ocr_pages: int = 0
    image_ocr_runs: int = 0


@dataclass
class ExtractedPdf:
    """Single concatenated document text with structured metadata for observability."""

    text: str
    metadata: dict[str, Any]
    stats: ExtractionStats = field(default_factory=ExtractionStats)


def _rows_to_markdown(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    clean: list[list[str]] = []
    for r in rows:
        row = [str(c).strip() if c is not None else "" for c in r]
        if any(cell for cell in row):
            clean.append(row)
    if not clean:
        return ""
    width = max(len(r) for r in clean)
    for r in clean:
        while len(r) < width:
            r.append("")
    lines: list[str] = []
    for r in clean:
        lines.append("| " + " | ".join(r) + " |")
    if len(clean) > 1:
        lines.insert(1, "| " + " | ".join("---" for _ in range(width)) + " |")
    return "\n".join(lines)


def _extract_tables_from_page(page: fitz.Page) -> tuple[list[str], int]:
    out: list[str] = []
    n = 0
    try:
        finder = page.find_tables()
        tables = getattr(finder, "tables", None) or []
    except (AttributeError, Exception):
        return out, 0
    for ti, tab in enumerate(tables, start=1):
        try:
            rows = tab.extract()
        except (AttributeError, Exception):
            continue
        if not rows:
            continue
        md = _rows_to_markdown([list(r) for r in rows if r is not None])
        if md:
            n += 1
            out.append(
                f"\n[Table {ti} — page {page.number + 1}]\n{md}\n"
            )
    return out, n


def _page_ocr(
    page: fitz.Page,
    settings: Settings,
) -> tuple[str, bool]:
    """Render page to an image and run Tesseract. Returns (text, success)."""
    if pytesseract is None or Image is None:
        return "", False
    try:
        m = fitz.Matrix(settings.pdf_ocr_zoom, settings.pdf_ocr_zoom)
        pix = page.get_pixmap(matrix=m, alpha=False)
        pil = Image.open(io.BytesIO(pix.tobytes("png")))
        if pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        return (pytesseract.image_to_string(pil) or "").strip(), True
    except Exception:
        return "", False


def _extract_image_ocr(
    _page: fitz.Page,
    doc: fitz.Document,
    settings: Settings,
    img_list: list,
    index: int,
) -> tuple[str, bool]:
    if not settings.pdf_ocr_images or pytesseract is None or Image is None:
        return "", False
    if index >= len(img_list):
        return "", False
    try:
        xref = img_list[index][0]
        b = doc.extract_image(xref)
        if not b or "image" not in b:
            return "", False
        pil = Image.open(io.BytesIO(b["image"]))
        if pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        t = pytesseract.image_to_string(pil) or ""
        t = t.strip()
        if t:
            return t, True
    except Exception:
        return "", False
    return "", False


def _normalize_body_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t) if t else t
    return t


def extract_pdf_bytes(
    pdf_bytes: bytes,
    file_name: str,
    settings: Settings,
) -> ExtractedPdf:
    """
    Extract a single document string suitable for RAG chunking.
    Per-page: native text, detected tables, optional full-page OCR, image OCR.
    """
    stats = ExtractionStats()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        stats.page_count = doc.page_count
        parts: list[str] = []
        for page in doc:
            pnum = page.number + 1
            header = f"\n\n## Page {pnum}\n\n"
            page_parts: list[str] = [header]
            page_has_content = False

            raw = page.get_text("text") or ""
            body = _normalize_body_text(raw)

            table_strs, tc = _extract_tables_from_page(page)
            stats.tables_extracted += tc
            for ts in table_strs:
                page_parts.append(ts)
            if table_strs:
                page_has_content = True

            img_list = page.get_images(full=True) or []
            stats.images_found += len(img_list)

            use_ocr = False
            if settings.pdf_ocr_enabled and pytesseract is not None:
                if len(body) < settings.pdf_ocr_min_text_chars or not body:
                    ocr_text, _ok = _page_ocr(page, settings)
                    if ocr_text and len(ocr_text) > len(body):
                        body = _normalize_body_text(ocr_text)
                        use_ocr = True
                        stats.ocr_pages += 1

            if body:
                page_parts.append("### Text\n" + body + "\n")
                page_has_content = True
            elif not use_ocr and not table_strs and settings.pdf_ocr_enabled and pytesseract:
                ocr_text, _ok = _page_ocr(page, settings)
                if ocr_text:
                    page_parts.append("### Text (OCR)\n" + ocr_text + "\n")
                    page_has_content = True
                    stats.ocr_pages += 1

            if settings.pdf_ocr_images and pytesseract is not None:
                for ii in range(len(img_list)):
                    t_img, did = _extract_image_ocr(
                        page, doc, settings, img_list, ii
                    )
                    if t_img and did:
                        page_parts.append(
                            f"### Image {ii + 1} (OCR)\n{t_img}\n"
                        )
                        page_has_content = True
                        stats.image_ocr_runs += 1

            if not page_has_content:
                page_parts.append("_(no extractable text on this page)_\n")
            parts.append("".join(page_parts))

        full = "".join(parts).strip()
    finally:
        doc.close()

    if not full or len(full) < 1:
        full = f"(empty PDF: {file_name})"

    metadata: dict[str, Any] = {
        "source": "pdf",
        "file_name": file_name,
        "modality": "multimodal_pdf",
        "page_count": stats.page_count,
        "tables_extracted": stats.tables_extracted,
        "images_embedded": stats.images_found,
        "ocr_page_count": stats.ocr_pages,
        "image_ocr_count": stats.image_ocr_runs,
    }
    return ExtractedPdf(text=full, metadata=metadata, stats=stats)
