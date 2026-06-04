"""Runtime settings with hot-reload support.

PUT /api/settings updates values in memory. The RAGEngine reference
is injected at app startup so changes take effect on the next query.
No restart needed.
"""

import os
from dataclasses import dataclass, asdict


@dataclass
class RuntimeSettings:
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"
    top_k: int = 5
    rerank_enabled: bool = True
    stream_enabled: bool = True
    chunk_size: int = 1024
    chunk_overlap: int = 200

    def __post_init__(self):
        if not self.llm_api_key:
            self.llm_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self._engine_ref = None  # set by AppState at startup

    def set_engine(self, engine):
        """Called once at app startup to bind the RAGEngine instance."""
        self._engine_ref = engine

    def apply(self, updates: dict) -> dict:
        for key in ("llm_api_key", "llm_model", "llm_base_url", "top_k",
                     "rerank_enabled", "stream_enabled", "chunk_size", "chunk_overlap"):
            if key in updates:
                setattr(self, key, updates[key])
        self._sync_to_engine()
        return self.to_dict()

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_engine_ref", None)
        # Mask API key
        k = d.get("llm_api_key", "")
        if k and len(k) > 8:
            d["llm_api_key"] = k[:8] + "****" + k[-4:]
        return d

    def _sync_to_engine(self):
        engine = self._engine_ref
        if engine is None:
            return
        engine._api_key = self.llm_api_key
        engine._model = self.llm_model
        engine._base_url = self.llm_base_url
        engine._enable_rerank = self.rerank_enabled
        engine._enable_stream = self.stream_enabled
