"""FastAPI application — lifespan, CORS, static files, singleton init."""

import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import Config
from core.vectordb import VectorStore, LocalEmbedder, DummyEmbedder, QueryCache
from core.retrieval import BM25Index, HybridRetriever, CrossEncoderReranker
from core.llm import RAGEngine
from server.settings_manager import RuntimeSettings
from server.routes import router as api_router


def _create_embedder(cfg: Config):
    try:
        return LocalEmbedder(model_name=cfg.embed_model, local_files_only=True)
    except Exception:
        pass
    if cfg.hf_endpoint:
        try:
            os.environ["HF_ENDPOINT"] = cfg.hf_endpoint
            return LocalEmbedder(model_name=cfg.embed_model)
        except Exception:
            pass
    return DummyEmbedder(dim=384)


def _rebuild_bm25(store, bm25):
    all_docs = []
    for src in store.list_sources():
        for i, c in enumerate(store.get_by_source(src)):
            meta = c.get("metadata", {})
            all_docs.append({
                "text": c["text"],
                "metadata": meta,
                "id": f"{src}_{meta.get('chunk_index', i)}",
            })
    bm25.build(all_docs)


def _create_reranker(cfg: Config):
    if not cfg.enable_rerank or not cfg.rerank_model:
        return None
    try:
        return CrossEncoderReranker(model_name=cfg.rerank_model)
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cfg = Config()
    os.makedirs(cfg.md_output_dir, exist_ok=True)

    embedder = _create_embedder(cfg)
    store = VectorStore(persist_dir=cfg.db_dir, collection_name=cfg.collection_name,
                        embedder=embedder)
    bm25 = BM25Index()
    _rebuild_bm25(store, bm25)

    hybrid = HybridRetriever(store, bm25)
    cache = QueryCache(max_size=cfg.cache_size, ttl=cfg.cache_ttl)
    reranker = _create_reranker(cfg)

    settings = RuntimeSettings()
    engine = RAGEngine(
        retriever=hybrid,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        reranker=reranker,
        cache=cache,
        enable_rewrite=cfg.enable_rewrite,
        enable_rerank=settings.rerank_enabled,
        enable_stream=settings.stream_enabled,
    )
    settings.set_engine(engine)

    # Store on app.state for route access
    app.state.rag_engine = engine
    app.state.vector_store = store
    app.state.bm25 = bm25
    app.state.settings = settings
    app.state.core_config = cfg

    print(f"  Server ready — {store.count()} chunks indexed")
    print(f"  Rerank: {'on' if settings.rerank_enabled else 'off'}")
    print(f"  Stream: {'on' if settings.stream_enabled else 'off'}")
    print(f"  API: http://0.0.0.0:8000/api/chat")

    yield
    print("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Knowledge Base", version="1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    return app


app = create_app()
