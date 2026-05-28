import os
from parsing.models import Document, Chunk
from parsing.chunker import chunk_document as _chunk_doc
from imageproc.processor import ImageProcessor


def parse_file(
    filepath: str,
    image_processor: ImageProcessor | None = None,
) -> Document:
    """Auto-detect file type and parse into a Document.

    For PDFs, if image_processor is provided, embedded images are extracted
    and processed (OCR + VLM), with descriptions inserted into the text.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        from parsing.pdf_parser import parse_pdf
        return parse_pdf(filepath, image_processor=image_processor)
    elif ext in (".md", ".markdown"):
        from parsing.markdown_parser import parse_markdown
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
    image_processor: ImageProcessor | None = None,
) -> tuple[Document, list[Chunk]]:
    """Parse and chunk a file in one call. Returns (document, chunks).

    For PDFs, pass image_processor to extract and describe embedded images.
    """
    doc = parse_file(filepath, image_processor=image_processor)
    chunks = chunk_document(doc, chunk_size, chunk_overlap)
    doc.chunks = chunks
    return doc, chunks
