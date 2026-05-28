import numpy as np
from vectordb.store import VectorStore
from vectordb.embeddings import BaseEmbedder


class Retriever:
    """Retrieval strategies on top of VectorStore."""

    def __init__(self, store: VectorStore):
        self._store = store

    @property
    def embedder(self) -> BaseEmbedder | None:
        return self._store._embedder

    def retrieve(
        self, query: str, top_k: int = 5, where: dict | None = None
    ) -> list[dict]:
        """Basic semantic similarity search."""
        return self._store.search(query, top_k=top_k, where=where)

    def retrieve_with_mmr(
        self,
        query: str,
        top_k: int = 5,
        fetch_factor: int = 3,
        lambda_mult: float = 0.5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Maximum Marginal Relevance — balances relevance and diversity.

        lambda_mult = 1: pure relevance (same as retrieve)
        lambda_mult = 0: pure diversity
        lambda_mult = 0.5: balanced (default)
        """
        if not self.embedder:
            # Fall back to basic search when no embedder is available
            return self.retrieve(query, top_k=top_k, where=where)

        fetch_k = top_k * fetch_factor
        candidates = self._store.search(query, top_k=fetch_k, where=where)

        if len(candidates) <= top_k:
            return candidates

        # Embed all candidate texts for MMR computation
        candidate_vectors = self.embedder.embed([c["text"] for c in candidates])
        query_vector = self.embedder.embed_query(query)

        # Cosine similarity: since vectors are normalized, use dot product
        similarities = np.dot(candidate_vectors, query_vector)

        selected_indices = []
        for _ in range(min(top_k, len(candidates))):
            mmr_scores = []
            for i, vec in enumerate(candidate_vectors):
                if i in selected_indices:
                    mmr_scores.append(-np.inf)
                    continue
                relevance = similarities[i]
                if selected_indices:
                    redundancy = max(
                        np.dot(candidate_vectors[i], candidate_vectors[j])
                        for j in selected_indices
                    )
                else:
                    redundancy = 0
                mmr_scores.append(lambda_mult * relevance - (1 - lambda_mult) * redundancy)
            best = int(np.argmax(mmr_scores))
            selected_indices.append(best)

        return [candidates[i] for i in selected_indices]

    def retrieve_with_filter(
        self, query: str, top_k: int = 5, where: dict | None = None
    ) -> list[dict]:
        """Search with ChromaDB metadata filter."""
        return self._store.search(query, top_k=top_k, where=where)
