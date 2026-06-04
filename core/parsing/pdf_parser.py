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
import shutil
import tempfile

from core.parsing.models import Document


def parse_pdf(filepath: str, md_output_dir: str | None = None,
              use_hybrid: bool = False) -> Document:
    """Parse a PDF file into a Document.

    Uses OpenDataLoader PDF (basic deterministic mode) to convert
    PDF → Markdown, then parses with the standard markdown parser.

    Set use_hybrid=True to enable the AI backend for OCR/VLM/table
    enhancement (requires a separately running hybrid server).
    """
    import opendataloader_pdf
    from core.parsing.markdown_parser import parse_markdown

    with tempfile.TemporaryDirectory() as tmpdir:
        kwargs = dict(
            input_path=[filepath],
            output_dir=tmpdir,
            format="markdown",
            image_output="embedded",
        )
        if use_hybrid:
            kwargs.update(hybrid="docling-fast", hybrid_mode="full")
        opendataloader_pdf.convert(**kwargs)

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

        # Save intermediate markdown for inspection
        if md_output_dir:
            os.makedirs(md_output_dir, exist_ok=True)
            dest = os.path.join(md_output_dir, f"{base}.md")
            shutil.copy2(md_path, dest)
            print(f"  Markdown saved: {dest}")

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
