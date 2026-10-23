"""
main.py — FastAPI Application Entry Point
------------------------------------------
Handles API routing, request validation, health monitoring,
and delegates query processing to the canonical core agent.

Endpoints:
  - POST /agent       : Main agentic query endpoint
  - POST /ask         : Legacy RAG endpoint (delegates to core agent)
  - POST /agent/image : Multi-modal vision analysis
  - POST /upload      : PDF ingestion endpoint
  - GET  /health      : System health check
  - GET  /workspace/* : Workspace session status
  - GET  /memory/*    : Memory management
  - GET  /monitor/*   : Performance monitoring
  - GET  /cache/*     : Query cache control
  - GET  /eval/*      : Evaluation report
"""

import os
import uuid
import json
import logging
import tempfile
from typing import Optional
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.models import AgentRequest, AgentResponse
from core.agent import run_agent
from core.workspace import get_or_create_workspace, get_workspace_manager
from rag.retrieve import get_retriever
from rag.cache import cache
from rag.monitor import monitor
from rag.memory import get_memory_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Secure AI Agent",
    description="Enterprise-grade Secure Agentic AI System with RAG, Tool Execution, and Memory",
    version="2.0.0"
)

memory = get_memory_manager()

@lru_cache(maxsize=1)
def get_cached_retriever():
    return get_retriever()


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Secure AI Agent",
        "model": settings.LLM_MODEL,
        "max_agent_steps": settings.MAX_AGENT_STEPS
    }


# ── Core Agent Endpoint ──────────────────────────────────────────────────────

@app.post("/agent")
def ask_agent(
    request: AgentRequest,
    x_user_role: Optional[str] = Header(default="student", alias="X-User-Role"),
    x_user_id: Optional[str] = Header(default="default_user", alias="X-User-Id")
):
    """
    🤖 Primary Canonical Agent Endpoint
    Executes security validation, memory recall, planning, dynamic tool selection,
    multi-step execution, reflection, and answer synthesis.
    """
    session_id = request.session_id or f"sess_{x_user_id}"
    user_role = request.user_role or x_user_role
    user_id = request.user_id or x_user_id

    # Check query cache
    cached = cache.get(request.question)
    if cached:
        logger.info(f"⚡ Cache HIT for query: {request.question[:50]}...")
        return cached

    result = run_agent(
        query=request.question,
        user_role=user_role,
        user_id=user_id,
        session_id=session_id
    )

    # Cache successful responses
    if result and result.get("search_type") != "security_blocked":
        cache.set(request.question, result)

    return result


# ── Legacy /ask Endpoint (Backwards Compatibility) ───────────────────────────

class LegacyQueryRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/ask")
def ask_question(request: LegacyQueryRequest):
    """
    Legacy asking endpoint. Automatically delegates to the core agent engine.
    """
    return ask_agent(AgentRequest(
        question=request.question,
        session_id=request.session_id
    ))


# ── Image Analysis Endpoint ──────────────────────────────────────────────────

@app.post("/agent/image")
async def ask_agent_image(
    question: str = Form(...),
    session_id: str = Form(default="default"),
    image: UploadFile = File(...)
):
    """
    🖼️ Multi-modal Vision Analysis
    """
    from tools.image_tool import run_image_tool

    image_bytes = await image.read()
    result = run_image_tool(query=question, image_input=image_bytes)

    return {
        "answer": result,
        "sources": ["Image Analysis Tool (Vision)"],
        "search_type": "image",
        "attempts": 1,
        "is_multi_step": False
    }


# ── Document Upload & Ingestion ───────────────────────────────────────────────

CHROMA_PATH = "chroma_db"
UPLOAD_DIR = "data/uploads"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), force_reindex: bool = Form(default=False)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"message": "❌ Only PDF files supported!"})

    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        return JSONResponse(status_code=400, content={"message": f"❌ File '{file.filename}' is empty (0 bytes)."})

    if file_size > 50 * 1024 * 1024:  # 50 MB limit
        return JSONResponse(status_code=400, content={"message": f"❌ File '{file.filename}' exceeds 50MB limit."})

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_path = os.path.join(UPLOAD_DIR, file.filename)

    if os.path.exists(saved_path) and not force_reindex:
        logger.info(f"ℹ️ Document '{file.filename}' already exists at {saved_path}")
        return {
            "message": f"ℹ️ '{file.filename}' is already indexed and ready.",
            "chunks": 0,
            "filename": file.filename,
            "file_size": file_size,
            "already_indexed": True
        }

    tmp_path = None
    try:
        logger.info(f"📄 Uploading PDF document: {file.filename} ({file_size} bytes)")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            return JSONResponse(status_code=400, content={"message": f"⚠️ No text could be extracted from '{file.filename}'."})

        for doc in documents:
            doc.metadata["source"] = file.filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        if not chunks:
            return JSONResponse(status_code=400, content={"message": f"⚠️ Document '{file.filename}' resulted in 0 text chunks."})

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        vectordb.add_documents(chunks)

        import shutil
        shutil.copy2(tmp_path, saved_path)

        get_cached_retriever.cache_clear()

        logger.info(f"✅ Ingested {file.filename}: {len(chunks)} chunks saved to {saved_path}")
        return {
            "message": f"✅ '{file.filename}' uploaded and indexed successfully!",
            "chunks": len(chunks),
            "filename": file.filename,
            "file_size": file_size,
            "saved_path": saved_path,
            "already_indexed": False
        }
    except Exception as e:
        logger.error(f"❌ Upload failed for {file.filename}: {e}")
        return JSONResponse(status_code=500, content={"message": f"❌ Upload failed: {str(e)}"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ── Workspace Endpoints ──────────────────────────────────────────────────────

@app.get("/workspace/status")
def workspace_status(session_id: str = "default"):
    ws = get_or_create_workspace(session_id)
    return ws.summary()

@app.get("/workspace/tasks/{task_id}")
def get_workspace_task(
    task_id: str,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    session_id: Optional[str] = None
):
    """
    Retrieve full task state by task ID.
    Enforces session isolation if session_id or X-User-Id header is supplied.
    """
    mgr = get_workspace_manager()
    verify_session = session_id or (f"sess_{x_user_id}" if x_user_id else None)
    try:
        ws = mgr.get_workspace(task_id, session_id=verify_session)
        if not ws:
            return JSONResponse(status_code=404, content={"message": f"Task '{task_id}' not found."})
        return ws.to_dict()
    except PermissionError as pe:
        return JSONResponse(status_code=403, content={"message": str(pe)})

@app.get("/workspace/sessions/{session_id}/tasks")
def list_session_tasks(session_id: str, limit: int = 20):
    """List all workspace tasks for a session."""
    mgr = get_workspace_manager()
    tasks = mgr.list_session_tasks(session_id, limit=limit)
    return {
        "session_id": session_id,
        "count": len(tasks),
        "tasks": [t.summary() for t in tasks]
    }

@app.delete("/workspace/sessions/{session_id}")
def clear_workspace_session(session_id: str):
    """Clear workspace tasks for a session."""
    mgr = get_workspace_manager()
    mgr.clear_session(session_id)
    return {"status": f"✅ Workspace cleared for session: {session_id}"}


# ── Memory Endpoints ─────────────────────────────────────────────────────────

@app.delete("/memory/clear")
def clear_memory(session_id: str = "default"):
    memory.clear(session_id)
    return {"status": f"✅ Memory cleared for session: {session_id}"}

@app.get("/memory/recent")
def recent_memory(session_id: str = "default"):
    recent = memory.get_recent(session_id)
    return {"session_id": session_id, "conversations": recent}


# ── Monitor & Observability Endpoints ────────────────────────────────────────

@app.get("/monitor/summary")
def monitor_summary():
    return monitor.summary()

@app.get("/monitor/recent")
def monitor_recent(n: int = 10):
    return {"recent": monitor.recent(n)}

@app.delete("/monitor/clear")
def monitor_clear():
    monitor.clear()
    return {"status": "✅ Monitor log cleared"}


# ── Cache Management Endpoints ───────────────────────────────────────────────

@app.get("/cache/stats")
def cache_stats():
    return cache.stats()

@app.delete("/cache/clear")
def cache_clear():
    cache.clear()
    return {"status": "✅ Cache cleared"}


# ── Evaluation Endpoints ─────────────────────────────────────────────────────

EVAL_REPORT_PATH = os.path.join("metrics", "eval_report.json")

@app.get("/eval/report")
def eval_report():
    if not os.path.exists(EVAL_REPORT_PATH):
        return JSONResponse(
            status_code=404,
            content={"message": "No eval report found. Run `python run_eval.py` first."}
        )
    try:
        with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        return report
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Failed to read eval report: {str(e)}"})