import os


class Config:
    """Unified configuration for the knowledge base Q&A system.

    Loaded from environment variables with sensible defaults.
    API keys are read from env vars only (never stored in code).
    """

    def __init__(self, **overrides):
        # Paths
        self.docs_dir = overrides.get("docs_dir", os.getenv("KB_DOCS_DIR", "./docs"))
        self.db_dir = overrides.get("db_dir", os.getenv("KB_DB_DIR", "./chroma_db"))
        self.collection_name = overrides.get("collection_name", os.getenv("KB_COLLECTION", "knowledge_base"))

        # HuggingFace
        self.hf_endpoint = overrides.get("hf_endpoint", os.getenv("HF_ENDPOINT", ""))

        # Embedding
        self.embed_model = overrides.get("embed_model", os.getenv("KB_EMBED_MODEL", "intfloat/multilingual-e5-small"))

        # LLM
        self.llm_api_key = overrides.get("llm_api_key", os.getenv("DEEPSEEK_API_KEY", ""))
        self.llm_model = overrides.get("llm_model", os.getenv("KB_LLM_MODEL", "deepseek-chat"))
        self.llm_base_url = overrides.get("llm_base_url", os.getenv("KB_LLM_BASE_URL", "https://api.deepseek.com"))

        # Chunking
        self.chunk_size = int(overrides.get("chunk_size", os.getenv("KB_CHUNK_SIZE", "1024")))
        self.chunk_overlap = int(overrides.get("chunk_overlap", os.getenv("KB_CHUNK_OVERLAP", "200")))

        # Retrieval
        self.top_k = int(overrides.get("top_k", os.getenv("KB_TOP_K", "5")))

        # Reranker
        self.rerank_model = overrides.get("rerank_model", os.getenv("KB_RERANK_MODEL", ""))
        self.enable_rerank = bool(self.rerank_model)

        # Query rewrite
        self.enable_rewrite = overrides.get("enable_rewrite", True)

        # OCR
        self.enable_ocr = overrides.get("enable_ocr", False)

        # Cache
        self.cache_size = int(overrides.get("cache_size", os.getenv("KB_CACHE_SIZE", "256")))
        self.cache_ttl = int(overrides.get("cache_ttl", os.getenv("KB_CACHE_TTL", "3600")))

    @property
    def llm_available(self) -> bool:
        return bool(self.llm_api_key)
