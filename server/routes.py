"""API routes — thin layer delegating to core modules."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from core.parsing.pipeline import process_file
from server.schemas import (
    ChatRequest, ChatResponse, DocumentInfo,
    SettingsResponse, SettingsUpdate, IngestResponse, DeleteResponse,
)

router = APIRouter()


def _get_core(request: Request):
    """Get core singletons from app state."""
    return (
        request.app.state.rag_engine,
        request.app.state.vector_store,
        request.app.state.bm25,
        request.app.state.settings,
    )


# ---- Chat ----

@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    engine, _, _, _ = _get_core(request)
    if req.stream:
        return StreamingResponse(
            _stream_chat(engine, req.question, req.top_k),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        result = engine.query(question=req.question, top_k=req.top_k)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=_friendly_error(e))


async def _stream_chat(engine, question: str, top_k: int):
    try:
        yield f"data: {json.dumps({'status': 'thinking'})}\n\n"
        for token in engine.query_stream(question=question, top_k=top_k):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': _friendly_error(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    if 'api_key' in msg.lower() or 'credential' in msg.lower():
        return '未配置 API Key，请在设置页面填入 DeepSeek API Key'
    if 'connection' in msg.lower() or 'connect' in msg.lower():
        return '无法连接 LLM 服务，请检查网络或 Base URL 设置'
    return msg


# ---- Ingest ----

@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...), request: Request = None):
    _, store, bm25, _ = _get_core(request)
    cfg = request.app.state.core_config
    indexed, total_chunks, errors = 0, 0, []

    for f in files:
        # Save uploaded file to temp location
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename or "")[1]) as tmp:
            tmp.write(await f.read())
            tmp_path = tmp.name

        try:
            doc, chunks = process_file(
                tmp_path,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                md_output_dir=cfg.md_output_dir,
            )
            # Override source_path to use original filename, not tmp path
            original_name = f.filename or os.path.basename(tmp_path)
            doc.source_path = original_name
            for c in chunks:
                c.metadata.source_path = original_name

            store.delete_by_source(doc.source_path)
            store.add_chunks(chunks)
            indexed += 1
            total_chunks += len(chunks)
        except Exception as e:
            errors.append(f"{f.filename}: {e}")
        finally:
            os.unlink(tmp_path)

    # Rebuild BM25
    _rebuild_bm25(store, bm25)

    return IngestResponse(indexed=indexed, total_chunks=total_chunks, errors=errors)


# ---- Documents ----

@router.get("/documents")
async def list_documents(request: Request):
    _, store, _, _ = _get_core(request)
    return [
        DocumentInfo(
            name=os.path.basename(src),
            source_path=src,
            chunks=len(store.get_by_source(src)),
        )
        for src in store.list_sources()
    ]


@router.delete("/documents/{name}")
async def delete_document(name: str, request: Request):
    _, store, bm25, _ = _get_core(request)
    # Resolve short name to full path
    target = None
    for src in store.list_sources():
        if os.path.basename(src) == name or src == name:
            target = src
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"Document not found: {name}")

    removed = store.delete_by_source(target)
    _rebuild_bm25(store, bm25)
    return DeleteResponse(status="deleted", chunks_removed=removed)


# ---- Settings ----

@router.get("/settings", response_model=SettingsResponse)
async def get_settings(request: Request):
    _, _, _, settings = _get_core(request)
    return SettingsResponse(**settings.to_dict())


@router.put("/settings", response_model=SettingsResponse)
async def update_settings(data: SettingsUpdate, request: Request):
    _, _, _, settings = _get_core(request)
    updates = {k: v for k, v in data.model_dump().items() if v is not None}
    applied = settings.apply(updates)
    return SettingsResponse(**applied)


# ---- helper ----

def _rebuild_bm25(store, bm25):
    """Rebuild BM25 index from vector store. Called after ingest/delete."""
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
