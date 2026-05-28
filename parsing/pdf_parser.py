import fitz
from parsing.models import Document
from imageproc.processor import ImageProcessor


def parse_pdf(
    filepath: str,
    image_processor: ImageProcessor | None = None,
) -> Document:
    """Extract structured text from a PDF, detecting headings by font heuristics.

    If image_processor is provided, embedded images are extracted and processed
    (OCR + VLM), with descriptions inserted into the text at image position.
    """
    doc = fitz.open(filepath)
    meta = doc.metadata
    text_blocks: list[str] = []
    sections: list[dict] = []
    current_section = {"title": "", "text": "", "page": 0}

    all_font_sizes = _collect_font_sizes(doc)
    body_size = _mode(all_font_sizes) if all_font_sizes else 11.0

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]

        # Pre-process all images on this page
        image_descriptions = _extract_page_images(doc, page, image_processor)
        img_idx = 0  # index into pre-processed descriptions

        for block in blocks:
            block_type = block.get("type", 0)

            if block_type == 1:  # image block
                if img_idx < len(image_descriptions):
                    desc = image_descriptions[img_idx]
                    img_idx += 1
                    if desc:
                        text_blocks.append(desc)
                        current_section["text"] += desc + "\n"
                continue

            if block_type != 0:  # skip other non-text blocks
                continue

            block_text, is_heading = _process_block(block, body_size)
            if not block_text:
                continue

            text_blocks.append(block_text)

            if is_heading:
                if current_section["text"] or current_section["title"]:
                    current_section["text"] = current_section["text"].strip()
                    sections.append(current_section)
                current_section = {"title": block_text, "text": "", "page": page_num}
            else:
                current_section["text"] += block_text + "\n"

    # Final section
    if current_section["text"] or current_section["title"]:
        current_section["text"] = current_section["text"].strip()
        sections.append(current_section)

    full_text = "\n\n".join(text_blocks)

    return Document(
        text=full_text.strip(),
        title=meta.get("title", ""),
        author=meta.get("author", ""),
        pages=doc.page_count,
        source_path=filepath,
        sections=sections,
    )


def _extract_page_images(
    doc: fitz.Document,
    page: fitz.Page,
    image_processor: ImageProcessor | None,
) -> list[str]:
    """Extract and process all images on a page. Returns list of descriptions."""
    if not image_processor:
        return []

    image_xrefs = [img[0] for img in page.get_images(full=True)]
    descriptions = []
    for xref in image_xrefs:
        try:
            img_data = doc.extract_image(xref)
            image_bytes = img_data.get("image")
            if image_bytes:
                desc = image_processor.process(image_bytes)
                if desc:
                    descriptions.append(desc)
        except Exception:
            pass
    return descriptions


# -- helpers (unchanged) --

def _collect_font_sizes(doc: fitz.Document) -> list[float]:
    sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(span["size"])
    return sizes


def _mode(values: list[float]) -> float:
    """Smallest frequent font size — body text is the smallest size used repeatedly."""
    from collections import Counter
    counts = Counter(values)
    frequent = [s for s, c in counts.items() if c >= 2]
    if frequent:
        return min(frequent)
    return min(values) if values else 11.0


def _process_block(block: dict, body_size: float) -> tuple[str, bool]:
    """Extract text from a block and determine if it's a heading."""
    lines = block.get("lines", [])
    if not lines:
        return "", False

    block_text_parts = []
    block_is_bold = None
    block_font_size = 0
    span_count = 0

    for line in lines:
        for span in line.get("spans", []):
            text = span["text"].strip()
            if text:
                block_text_parts.append(text)
                block_font_size += span["size"]
                span_count += 1
                is_bold = bool(span["flags"] & 2)
                block_is_bold = is_bold if block_is_bold is None else block_is_bold

    if span_count == 0:
        return "", False

    avg_size = block_font_size / span_count
    block_text = " ".join(block_text_parts)
    text_len = len(block_text)

    is_heading = (
        avg_size >= body_size * 1.3 and text_len < 300
    ) or (
        avg_size >= body_size * 1.15 and (block_is_bold or text_len < 80)
    )

    return block_text, is_heading
