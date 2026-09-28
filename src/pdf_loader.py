from pathlib import Path

import pymupdf


def read_pdf(pdf_path: str | Path) -> list[dict[str, str | int]]:
    """Extract text from a PDF, keeping each page's number."""
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"page": page_number, "text": text})

    return pages