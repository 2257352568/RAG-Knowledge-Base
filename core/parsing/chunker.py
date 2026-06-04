from core.parsing.models import Chunk, ChunkMetadata, Document

SEPARATORS = ["\n\n", "\n", ". ", "。", " "]


def split_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 200) -> list[str]:
    """Recursively split text by separator priority, keeping semantic units intact."""
    if not text:
        return []

    splits = _split_recursive(text, SEPARATORS, chunk_size)

    # Merge short splits and apply overlap
    chunks = _merge_splits(splits, chunk_size)
    chunks = _apply_overlap(chunks, chunk_size, chunk_overlap)
    return chunks


def _split_recursive(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Try each separator; split oversized pieces; recurse on what remains."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in separators:
        if sep in text:
            pieces = text.split(sep)
            result = []
            for piece in pieces:
                if piece.strip():
                    result.extend(_split_recursive(piece, separators, chunk_size))
            return result

    # Last resort: character-level split
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _merge_splits(splits: list[str], chunk_size: int) -> list[str]:
    """Merge adjacent splits that together still fit within chunk_size."""
    if not splits:
        return []
    merged = []
    buf = splits[0]
    for part in splits[1:]:
        if len(buf) + len(part) <= chunk_size:
            buf += part
        else:
            merged.append(buf)
            buf = part
    merged.append(buf)
    return merged


def _apply_overlap(chunks: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Extend each chunk with the beginning of the next for context overlap."""
    if not chunks or overlap <= 0:
        return chunks

    overlapped = []
    for i, chunk in enumerate(chunks):
        if i < len(chunks) - 1:
            next_start = chunks[i + 1][:overlap]
            # Only prepend previous chunk's tail if not the first
            if i > 0:
                prev_tail = chunks[i - 1][-overlap:]
                chunk = prev_tail + chunk
            # Extend with next chunk's head, clamped to chunk_size + overlap
            extended = chunk + next_start
            overlapped.append(extended[: chunk_size + overlap])
        else:
            if i > 0:
                chunk = chunks[i - 1][-overlap:] + chunk
            overlapped.append(chunk[: chunk_size + overlap])
    return overlapped


def chunk_document(
    doc: Document,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split document into chunks, respecting section boundaries as hard splits."""
    if doc.sections:
        chunks = _structure_aware_split(doc, chunk_size, chunk_overlap)
    else:
        splits = split_text(doc.text, chunk_size, chunk_overlap)
        chunks = _build_chunks(splits, doc.source_path)

    for i, chunk in enumerate(chunks):
        chunk.metadata.chunk_index = i
        chunk.metadata.total_chunks = len(chunks)
    return chunks


def _structure_aware_split(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split each section independently, preserving section boundaries."""
    all_chunks = []
    for section in doc.sections:
        title = section.get("title", "")
        body = section.get("text", "")
        page = section.get("page")

        if len(body) <= chunk_size:
            all_chunks.append(
                Chunk(
                    text=f"{title}\n{body}".strip() if title else body,
                    metadata=ChunkMetadata(
                        source_path=doc.source_path,
                        page_start=page,
                        page_end=page,
                        section_title=title,
                    ),
                )
            )
        else:
            # Prefix each chunk with the section title for context
            splits = split_text(body, chunk_size - len(title) - 2, chunk_overlap)
            for s in splits:
                text = f"{title}\n{s}" if title else s
                all_chunks.append(
                    Chunk(
                        text=text,
                        metadata=ChunkMetadata(
                            source_path=doc.source_path,
                            page_start=page,
                            page_end=page,
                            section_title=title,
                        ),
                    )
                )
    return all_chunks


def _build_chunks(splits: list[str], source_path: str) -> list[Chunk]:
    return [
        Chunk(
            text=s,
            metadata=ChunkMetadata(source_path=source_path),
        )
        for s in splits
    ]
