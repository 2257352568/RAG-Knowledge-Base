import chromadb
from chromadb.config import Settings
from parsing.models import Chunk
from vectordb.embeddings import BaseEmbedder


class VectorStore:
    """ChromaDB-backed vector store for RAG knowledge base."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "knowledge_base",
        embedder: BaseEmbedder | None = None,
    ):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedder = embedder
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and store a list of Chunks. Returns number of chunks added."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadatas = [_sanitize_metadata(c.metadata.to_dict()) for c in chunks]
        ids = self._make_ids(chunks)

        if self._embedder:
            embeddings = self._embedder.embed(texts)
            self._collection.add(
                ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
            )
        else:
            self._collection.add(ids=ids, documents=texts, metadatas=metadatas)

        return len(chunks)

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Insert or update chunks by ID. For incremental indexing."""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        metadatas = [_sanitize_metadata(c.metadata.to_dict()) for c in chunks]
        ids = self._make_ids(chunks)
        if self._embedder:
            embeddings = self._embedder.embed(texts)
            self._collection.upsert(
                ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
            )
        else:
            self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(chunks)

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> int:
        """Add raw texts with optional metadata. Returns count added."""
        if not texts:
            return 0

        metadatas = metadatas or [{}] * len(texts)
        metadatas = [_sanitize_metadata(m) for m in metadatas]
        ids = [f"raw_{i}_{hash(t)[:12]}" for i, t in enumerate(texts)]

        if self._embedder:
            embeddings = self._embedder.embed(texts)
            self._collection.add(
                ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
            )
        else:
            self._collection.add(ids=ids, documents=texts, metadatas=metadatas)

        return len(texts)

    def search(
        self, query: str, top_k: int = 5, where: dict | None = None
    ) -> list[dict]:
        """Semantic search. Returns list of {text, metadata, score}."""
        if self._embedder:
            query_embedding = self._embedder.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        return self._format_results(results)

    def delete_by_source(self, source_path: str) -> int:
        """Delete all chunks from a given source file. Returns count deleted."""
        results = self._collection.get(where={"source_path": source_path})
        ids_to_delete = results["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def count(self) -> int:
        return self._collection.count()

    def get_by_source(self, source_path: str) -> list[dict]:
        """Retrieve all chunks from a given source."""
        results = self._collection.get(where={"source_path": source_path})
        if not results["ids"]:
            return []
        return [
            {"text": d, "metadata": m}
            for d, m in zip(results["documents"], results["metadatas"])
        ]

    def list_sources(self) -> list[str]:
        """Return unique source paths in the collection."""
        if self.count() == 0:
            return []
        all_data = self._collection.get(include=["metadatas"])
        sources = set()
        for m in all_data["metadatas"]:
            if m and "source_path" in m:
                sources.add(m["source_path"])
        return sorted(sources)

    def _make_ids(self, chunks: list[Chunk]) -> list[str]:
        return [
            f"{c.metadata.source_path}_{c.metadata.chunk_index}"
            for c in chunks
        ]

    def _format_results(self, raw: dict) -> list[dict]:
        results = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            results.append({
                "id": doc_id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "score": round(1 - dists[i], 4) if i < len(dists) else 0,
            })
        return results


def _sanitize_metadata(meta: dict) -> dict:
    """Remove None values that ChromaDB cannot store."""
    return {
        k: v for k, v in meta.items()
        if v is not None and not isinstance(v, (list, dict))
    }
