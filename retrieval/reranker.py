class CrossEncoderReranker:
    """Cross-Encoder reranker for fine-grained relevance scoring.

    Bi-Encoder (vector search): encodes query & doc independently → fast, less accurate.
    Cross-Encoder: encodes (query, doc) jointly → slow, more accurate.

    Best practice: coarse recall (BM25 + vector) → fine rerank (Cross-Encoder).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)
        self._loaded = True

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank candidates by relevance to query.

        Each candidate must have a 'text' key. Returns top_k reranked results.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach cross-encoder score
        for i, score in enumerate(scores):
            candidates[i]["ce_score"] = round(float(score), 4)

        ranked = sorted(candidates, key=lambda x: x.get("ce_score", 0), reverse=True)
        return ranked[:top_k]


class DummyReranker:
    """No-op reranker for testing without downloading the model."""

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        # Pass through, mark as not reranked
        for c in candidates:
            c["ce_score"] = c.get("score", 0)
        return candidates[:top_k]
