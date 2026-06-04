#!/usr/bin/env python3
"""Personal knowledge base Q&A system.

Usage:
    python main.py ingest <file_or_dir>    # Index documents
    python main.py chat                     # Interactive Q&A
    python main.py status                   # Show indexed documents
    python main.py remove <source_path>    # Delete a document from index
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.parsing.pipeline import process_file
from core.vectordb import VectorStore, LocalEmbedder, DummyEmbedder, QueryCache
from core.vectordb.embeddings import BaseEmbedder
from core.retrieval import BM25Index, HybridRetriever
from core.llm import RAGEngine

# Color helpers for terminal output
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _create_embedder(cfg: Config) -> BaseEmbedder:
    """Create embedder, loading from cache first, falling back to mirror download."""
    import os as _os

    # 1. Try loading from local cache (no network needed)
    try:
        embedder = LocalEmbedder(model_name=cfg.embed_model, local_files_only=True)
        print(f"{_GREEN}  -> {cfg.embed_model} ready ({embedder.dim}-dimensional) [cached]{_RESET}")
        return embedder
    except Exception:
        pass

    # 2. Model not cached — try downloading via mirror
    mirrors = [cfg.hf_endpoint] if cfg.hf_endpoint else [
        "https://hf-mirror.com",
    ]

    last_error = None
    for mirror in mirrors:
        try:
            _os.environ["HF_ENDPOINT"] = mirror
            embedder = LocalEmbedder(model_name=cfg.embed_model)
            print(f"{_GREEN}  -> {cfg.embed_model} ready ({embedder.dim}-dimensional) [mirror]{_RESET}")
            return embedder
        except Exception as e:
            last_error = e
            continue

    print(f"{_YELLOW}  -> Model download failed: {last_error}{_RESET}")
    print(f"{_YELLOW}  -> Falling back to DummyEmbedder (BM25 still works, semantic search will not){_RESET}")
    return DummyEmbedder(dim=384)


def _rebuild_bm25(store: VectorStore, bm25: BM25Index) -> int:
    """Rebuild BM25 index from all chunks in the persistent VectorStore."""
    sources = store.list_sources()
    all_docs = []
    for src in sources:
        chunks = store.get_by_source(src)
        for i, c in enumerate(chunks):
            meta = c.get("metadata", {})
            doc_id = f"{src}_{meta.get('chunk_index', i)}"
            all_docs.append({"text": c["text"], "metadata": meta, "id": doc_id})
    bm25.build(all_docs)
    return len(all_docs)


def _build_rag_engine(cfg: Config):
    """Initialize all components and return a ready-to-use RAGEngine."""
    print(f"{_CYAN}Loading embedding model: {cfg.embed_model}{_RESET}")
    embedder = _create_embedder(cfg)

    print(f"{_CYAN}Opening vector store: {cfg.db_dir}{_RESET}")
    store = VectorStore(
        persist_dir=cfg.db_dir,
        collection_name=cfg.collection_name,
        embedder=embedder,
    )

    print(f"{_CYAN}Rebuilding BM25 index...{_RESET}")
    bm25 = BM25Index()
    n = _rebuild_bm25(store, bm25)
    print(f"{_GREEN}  -> {n} chunks from {len(store.list_sources())} documents{_RESET}")

    hybrid = HybridRetriever(store, bm25)
    cache = QueryCache(max_size=cfg.cache_size, ttl=cfg.cache_ttl)

    # Create reranker if enabled
    reranker = None
    if cfg.enable_rerank and cfg.rerank_model:
        from core.retrieval.reranker import CrossEncoderReranker
        try:
            reranker = CrossEncoderReranker(model_name=cfg.rerank_model)
            print(f"{_GREEN}  -> Reranker: {cfg.rerank_model}{_RESET}")
        except Exception as e:
            print(f"{_YELLOW}  -> Reranker unavailable ({e}), proceeding without{_RESET}")

    engine = RAGEngine(
        retriever=hybrid,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        reranker=reranker,
        cache=cache,
        enable_rewrite=cfg.enable_rewrite,
        enable_rerank=cfg.enable_rerank,
    )
    return engine


def cmd_ingest(cfg: Config, path: str):
    """Index a file or directory into the knowledge base."""
    full = os.path.abspath(path)

    # Collect files
    files = []
    if os.path.isfile(full):
        files.append(full)
    elif os.path.isdir(full):
        for root, _, filenames in os.walk(full):
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in (".pdf", ".md", ".markdown"):
                    files.append(os.path.join(root, f))
    else:
        print(f"{_RED}Error: path not found: {path}{_RESET}")
        sys.exit(1)

    if not files:
        print(f"{_YELLOW}No .pdf or .md files found in: {path}{_RESET}")
        return

    # Initialize store and embedder
    print(f"{_CYAN}Loading embedding model: {cfg.embed_model}{_RESET}")
    embedder = _create_embedder(cfg)

    store = VectorStore(
        persist_dir=cfg.db_dir,
        collection_name=cfg.collection_name,
        embedder=embedder,
    )

    # Ensure markdown output directory exists
    os.makedirs(cfg.md_output_dir, exist_ok=True)

    total_chunks = 0
    for fpath in files:
        print(f"\n{_BOLD}Ingesting: {fpath}{_RESET}")
        try:
            doc, chunks = process_file(
                fpath,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                md_output_dir=cfg.md_output_dir,
            )
            # Dedup: remove existing chunks for this source
            removed = store.delete_by_source(doc.source_path)
            if removed:
                print(f"  Removed {removed} existing chunks")

            added = store.add_chunks(chunks)
            total_chunks += added
            print(f"  {_GREEN}Indexed {added} chunks ({doc.pages or len(doc.sections)} p/sections){_RESET}")

        except Exception as e:
            print(f"  {_RED}Failed: {e}{_RESET}")

    print(f"\n{_GREEN}{_BOLD}Done. {total_chunks} total chunks indexed.{_RESET}")


def cmd_chat(cfg: Config):
    """Interactive Q&A session with streaming responses."""
    if not cfg.llm_available:
        print(f"{_RED}Error: DEEPSEEK_API_KEY not set.{_RESET}")
        print("  export DEEPSEEK_API_KEY=sk-...")
        sys.exit(1)

    print(f"{_BOLD}{_CYAN}{'=' * 50}{_RESET}")
    print(f"{_BOLD}{_CYAN}  Personal Knowledge Base Q&A{_RESET}")
    print(f"{_BOLD}{_CYAN}{'=' * 50}{_RESET}\n")

    engine = _build_rag_engine(cfg)

    print(f"\n{_YELLOW}Commands: /exit | /sources | /clear{_RESET}")
    print(f"{_YELLOW}Ask anything about your documents.{_RESET}\n")

    while True:
        try:
            question = input(f"{_BOLD}You: {_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        # Special commands
        if question == "/exit":
            print("Goodbye.")
            break
        elif question == "/sources":
            sources = engine.list_sources()
            if sources:
                print(f"\n{_CYAN}Indexed documents ({len(sources)}):{_RESET}")
                for s in sources:
                    print(f"  - {s}")
            else:
                print(f"\n{_YELLOW}No documents indexed yet.{_RESET}")
            print()
            continue
        elif question == "/clear":
            engine.clear_cache()
            print(f"{_GREEN}Cache cleared.{_RESET}\n")
            continue

        # RAG query with streaming
        print(f"\n{_CYAN}Assistant: {_RESET}", end="", flush=True)
        full_answer = []
        try:
            for chunk in engine.query_stream(
                question,
                top_k=cfg.top_k,
                temperature=0.7,
                max_tokens=2048,
            ):
                print(chunk, end="", flush=True)
                full_answer.append(chunk)
            print("\n")
        except Exception as e:
            print(f"\n{_RED}Error: {e}{_RESET}\n")


def cmd_status(cfg: Config):
    """Show indexed documents and statistics."""
    print(f"{_CYAN}Vector store: {cfg.db_dir}{_RESET}")
    embedder = _create_embedder(cfg)
    store = VectorStore(
        persist_dir=cfg.db_dir,
        collection_name=cfg.collection_name,
        embedder=embedder,
    )

    total = store.count()
    sources = store.list_sources()

    print(f"{_BOLD}Total chunks: {total}{_RESET}")
    print(f"{_BOLD}Documents: {len(sources)}{_RESET}\n")

    if sources:
        print(f"{_CYAN}Indexed documents:{_RESET}")
        for s in sources:
            chunks = store.get_by_source(s)
            print(f"  - {s}  ({len(chunks)} chunks)")
    else:
        print(f"{_YELLOW}No documents indexed.{_RESET}")
        print(f"  Run: python main.py ingest <file_or_directory>")


def cmd_remove(cfg: Config, path: str):
    """Remove a document from the knowledge base."""
    full = os.path.abspath(path)
    embedder = _create_embedder(cfg)
    store = VectorStore(
        persist_dir=cfg.db_dir,
        collection_name=cfg.collection_name,
        embedder=embedder,
    )

    removed = store.delete_by_source(full)
    if removed:
        print(f"{_GREEN}Removed {removed} chunks for: {full}{_RESET}")
    else:
        print(f"{_YELLOW}No chunks found for: {full}{_RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Personal Knowledge Base Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest ./docs/
  python main.py ingest paper.pdf
  python main.py chat
  python main.py status
  python main.py remove paper.pdf
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index a file or directory")
    p_ingest.add_argument("path", help="File or directory path to ingest")

    # chat
    sub.add_parser("chat", help="Start interactive Q&A session")

    # status
    sub.add_parser("status", help="Show indexed documents")

    # remove
    p_remove = sub.add_parser("remove", help="Remove a document from the index")
    p_remove.add_argument("path", help="Source path of the document to remove")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cfg = Config()

    if args.command == "ingest":
        cmd_ingest(cfg, args.path)
    elif args.command == "chat":
        cmd_chat(cfg)
    elif args.command == "status":
        cmd_status(cfg)
    elif args.command == "remove":
        cmd_remove(cfg, args.path)


if __name__ == "__main__":
    main()
