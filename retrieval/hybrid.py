from vectordb.store import VectorStore
from retrieval.bm25 import BM25Index


class HybridRetriever:
    """Hybrid retrieval: vector (semantic) + BM25 (keyword) with RRF fusion.

    RRF (Reciprocal Rank Fusion):
        RRF_score(d) = Σ 1/(k + rank_i(d))
        k=60 is the classic constant. No score normalization needed.
    """

    RRF_K = 60

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self._vs = vector_store
        self._bm25 = bm25_index

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        where: dict | None = None,
    ) -> list[dict]:
        """Hybrid search with RRF fusion. alpha: vector weight (0-1)."""
        fetch_k = max(top_k * 2, 10)
        vec_results = self._vs.search(query, top_k=fetch_k, where=where)
        bm25_results = self._bm25.search(query, top_k=fetch_k)

        if not vec_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return vec_results[:top_k]

        fused = self._rrf_fusion(vec_results, bm25_results, alpha)
        return fused[:top_k]

    def _rrf_fusion(
        self, vec_results: list[dict], bm25_results: list[dict], alpha: float
    ) -> list[dict]:
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, doc in enumerate(vec_results):
            doc_id = doc.get("id", f"vec_{rank}")
            scores[doc_id] = scores.get(doc_id, 0) + alpha / (self.RRF_K + rank + 1)
            docs[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc.get("id", f"bm25_{rank}")
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) / (self.RRF_K + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [docs[doc_id] | {"score": round(rrf_score, 4)}
                for doc_id, rrf_score in ranked]
