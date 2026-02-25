"""Extract text from PDFs using pdfplumber."""

import pdfplumber
from pathlib import Path


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Extract text from each page of a PDF.

    Returns a list of dicts with keys: text, page, source.
    """
    pdf_path = Path(pdf_path)
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({
                    "text": text,
                    "page": i + 1,
                    "source": pdf_path.name,
                })
    return pages
