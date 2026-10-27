"""Extract page-marked text from an RFP PDF.

Contestant models and the judges consume text, not PDFs — only the Claude
teacher reads the PDF directly. Page markers keep the schema's `page` field
answerable from text alone.

Usage: python src/pdftext.py data/rfps/HHS0015558.pdf   # writes .txt beside it
"""

import sys
from pathlib import Path

from pypdf import PdfReader


def pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(f"[page {i}]\n{text}")
    return "\n\n".join(pages)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pdf_path = Path(sys.argv[1])
    out = pdf_path.with_suffix(".txt")
    out.write_text(pdf_to_text(pdf_path))
    print(f"{pdf_path.name}: {len(PdfReader(pdf_path).pages)} pages -> {out.name}")
