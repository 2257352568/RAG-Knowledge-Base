import hashlib
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract embedding backend — swap between local models and remote APIs."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of texts to embedding vectors."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (default: delegate to embed)."""
        return self.embed([text])[0]


class DummyEmbedder(BaseEmbedder):
    """Deterministic hash-based embedder for offline testing.

    Uses SHA-256 hashing to produce consistent, semi-distinctive vectors per text.
    NOT for production — vectors capture word-level overlap, not semantics.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        vectors = []
        for text in texts:
            # Seed numpy with a hash of the text for deterministic output
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)  # normalize
            vectors.append(vec.tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self._dim


class LocalEmbedder(BaseEmbedder):
    """Sentence-transformers model running locally (no API calls, no cost)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", local_files_only: bool = False):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, local_files_only=local_files_only)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dim(self) -> int:
        return self._dim


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI-compatible embedding API (text-embedding-3-small, etc.)."""

    def __init__(
        self,
        api_key: str = "sk-xxx",
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
    ):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim = 1536 if "small" in model else 3072

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    @property
    def dim(self) -> int:
        return self._dim


class DeepSeekEmbedder(OpenAIEmbedder):
    """DeepSeek Embedding API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str = "sk-xxx",
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.deepseek.com",
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model)
