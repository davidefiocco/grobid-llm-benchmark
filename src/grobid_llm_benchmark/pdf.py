"""Render PDF pages to PNG images for multimodal LLM input."""

from __future__ import annotations

import base64
import warnings
from pathlib import Path

import fitz  # PyMuPDF


def render_pages_to_png(
    pdf_path: Path,
    max_pages: int | None = None,
    dpi: int = 150,
) -> list[bytes]:
    """Render (up to ``max_pages``) pages of ``pdf_path`` to PNG bytes.

    Header metadata lives on the first page(s); references live on the last page(s).
    When ``max_pages`` is set and the PDF is longer, we take the first and last pages so
    both header and bibliography are covered without sending the whole document.
    ``max_pages`` of ``None`` or ``<= 0`` renders **all** pages (whole-document coverage,
    needed for parity with GROBID on the fulltext structures spread across the body).
    """
    if max_pages is not None and max_pages <= 0:
        max_pages = None
    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        if max_pages is None or n <= max_pages:
            indices = list(range(n))
        else:
            head = (max_pages + 1) // 2
            tail = max_pages - head
            indices = list(range(head)) + list(range(n - tail, n))
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        images: list[bytes] = []
        for i in indices:
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix)
            images.append(pix.tobytes("png"))
        return images
    finally:
        doc.close()


def render_pages_to_base64(
    pdf_path: Path,
    max_pages: int | None = None,
    dpi: int = 150,
) -> list[str]:
    """Same as :func:`render_pages_to_png` but base64-encoded (for JSON/HTTP APIs)."""
    return [
        base64.b64encode(png).decode("ascii")
        for png in render_pages_to_png(pdf_path, max_pages, dpi)
    ]


def find_pdf(article_dir: Path) -> Path | None:
    """Return the (deterministically first) PDF in an article directory, if any."""
    pdfs = sorted(p for p in Path(article_dir).iterdir() if p.suffix.lower() == ".pdf")
    if len(pdfs) > 1:
        warnings.warn(f"{article_dir} has {len(pdfs)} PDFs; using {pdfs[0].name}", stacklevel=2)
    return pdfs[0] if pdfs else None


def find_pdfs(data_dir: Path) -> list[Path]:
    """Return every PDF under ``data_dir`` (recursively), sorted.

    Gold-agnostic: PDFs are returned whether or not a gold ``.nxml`` sits beside them.
    """
    return sorted(p for p in Path(data_dir).rglob("*.pdf") if p.is_file())


def extract_text(pdf_path: Path, max_pages: int | None = None) -> str:
    """Extract the raw text layer of the PDF (optional context for text-only backends).

    ``max_pages`` of ``None`` or ``<= 0`` reads the whole document.
    """
    if max_pages is not None and max_pages <= 0:
        max_pages = None
    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
        return "\n".join(doc.load_page(i).get_text() for i in range(n))
    finally:
        doc.close()
