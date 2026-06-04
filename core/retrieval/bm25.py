import math
import jieba


class BM25Index:
    """BM25 keyword search index for hybrid retrieval.

    BM25(D, Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1×(1-b+b×|D|/avgdl))

    k1=1.5, b=0.75 are the classic Okapi BM25 parameters.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._documents: list[dict] = []        # original docs with metadata
        self._doc_tokens: list[list[str]] = []   # tokenized documents
        self._doc_freq: dict[str, int] = {}       # DF: number of docs containing each term
        self._avgdl: float = 0.0                  # average document length
        self._idf: dict[str, float] = {}          # precomputed IDF

    def build(self, documents: list[dict]):
        """Index a list of documents, each with at least {'text': str}."""
        self._documents = documents
        self._doc_tokens = []
        self._doc_freq = {}
        total_len = 0

        for doc in documents:
            tokens = _tokenize(doc["text"])
            self._doc_tokens.append(tokens)
            total_len += len(tokens)
            seen = set()
            for t in tokens:
                if t not in seen:
                    self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
                    seen.add(t)

        N = len(documents) or 1
        self._avgdl = total_len / N
        # Precompute IDF
        for term, df in self._doc_freq.items():
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def add(self, documents: list[dict]):
        """Incrementally add documents (rebuilds index)."""
        all_docs = self._documents + documents
        self.build(all_docs)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 search. Returns list of {text, metadata, score}."""
        if not self._documents:
            return []

        query_tokens = _tokenize(query)
        scores = [self._score(i, query_tokens) for i in range(len(self._documents))]

        # Sort by score descending
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        return [
            {
                "text": self._documents[i].get("text", ""),
                "metadata": self._documents[i].get("metadata", {}),
                "score": round(score, 4),
                "id": self._documents[i].get("id", f"bm25_{i}"),
            }
            for i, score in ranked[:top_k] if score > 0
        ]

    def _score(self, doc_idx: int, query_tokens: list[str]) -> float:
        doc_tokens = self._doc_tokens[doc_idx]
        doc_len = len(doc_tokens)
        score = 0.0

        for token in query_tokens:
            idf = self._idf.get(token, 0)
            if idf == 0:
                continue
            tf = doc_tokens.count(token)
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (1 - self._b + self._b * doc_len / self._avgdl)
            score += idf * numerator / denominator

        return score


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing. jieba for Chinese + whitespace for English."""
    tokens = []
    for word in jieba.cut(text):
        word = word.strip().lower()
        if word and len(word) > 1:
            tokens.append(word)
    return tokens
