from dataclasses import dataclass, field


@dataclass
class ChunkMetadata:
    source_path: str
    page_start: int | None = None
    page_end: int | None = None
    section_title: str = ""
    chunk_index: int = 0
    total_chunks: int = 0

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata


@dataclass
class Document:
    text: str
    title: str = ""
    author: str = ""
    pages: int = 0
    source_path: str = ""
    sections: list[dict] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
