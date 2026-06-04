"""PDF parser using OpenDataLoader PDF.

OpenDataLoader PDF converts PDFs to Markdown with:
  - XY-Cut++ reading order (handles multi-column layouts)
  - Table extraction (bordered and borderless)
  - Heading detection
  - Formula → LaTeX
  - 80+ language OCR support

The output Markdown is fed into our existing markdown parser,
so all downstream chunking and retrieval work identically.
"""

import os
import tempfile

from core.parsing.models import Document


def parse_pdf(filepath: str) -> Document:
    """Parse a PDF file into a Document.

    Uses OpenDataLoader PDF to convert PDF → Markdown, then
    parses the markdown with our standard markdown parser.
    """
    import opendataloader_pdf
    from core.parsing.markdown_parser import parse_markdown

    with tempfile.TemporaryDirectory() as tmpdir:
        opendataloader_pdf.convert(
            input_path=[filepath],
            output_dir=tmpdir,
            format="markdown",
        )

        # Find the output markdown file
        base = os.path.splitext(os.path.basename(filepath))[0]
        md_path = os.path.join(tmpdir, f"{base}.md")

        if not os.path.isfile(md_path):
            # Try globbing
            for f in os.listdir(tmpdir):
                if f.endswith(".md"):
                    md_path = os.path.join(tmpdir, f)
                    break
            else:
                raise RuntimeError(
                    f"OpenDataLoader did not produce output for {filepath}"
                )

        doc = parse_markdown(md_path)

    # Override source_path and page count
    doc.source_path = filepath
    try:
        import fitz  # still use PyMuPDF just for metadata
        with fitz.open(filepath) as pdf:
            doc.pages = pdf.page_count
            meta = pdf.metadata
            if meta:
                doc.title = meta.get("title", "") or doc.title
                doc.author = meta.get("author", "") or doc.author
    except Exception:
        pass

    return doc
