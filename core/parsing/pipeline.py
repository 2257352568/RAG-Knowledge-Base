"""Parse → chunk pipeline. Entry point for all document ingestion."""

import os

from core.parsing.models import Document, Chunk
from core.parsing.chunker import chunk_document as _chunk_doc


def parse_file(filepath: str, md_output_dir: str | None = None) -> Document:
    """Auto-detect file type and parse into a Document.

    PDFs are converted to Markdown by OpenDataLoader PDF,
    then parsed by our standard Markdown parser.
    If md_output_dir is set, intermediate .md files are saved there.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        from core.parsing.pdf_parser import parse_pdf
        return parse_pdf(filepath, md_output_dir=md_output_dir)
    elif ext in (".md", ".markdown"):
        from core.parsing.markdown_parser import parse_markdown
        return parse_markdown(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_document(
    doc: Document,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split a Document into chunks."""
    return _chunk_doc(doc, chunk_size, chunk_overlap)


def process_file(
    filepath: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    md_output_dir: str | None = None,
) -> tuple[Document, list[Chunk]]:
    """Parse and chunk a file in one call. Returns (document, chunks)."""
    doc = parse_file(filepath, md_output_dir=md_output_dir)
    chunks = chunk_document(doc, chunk_size, chunk_overlap)
    doc.chunks = chunks
    return doc, chunks
