"""
streaming_rag.py
----------------
FastAPI SSE Streaming + Confidence Score + Disclaimer system.

Features:
  - Server-Sent Events (SSE) streaming responses
  - Confidence score (0-100%) with High/Medium/Low label
  - Source file name in response
  - Auto disclaimer for low confidence answers
  - Integrates with ChromaDB + domain_classifier + role_access
  - ⏱️ Latency tracking (retrieval, reranking, generation, TTFT)

Run:
    uvicorn streaming_rag:app --reload --port 8001

Test SSE stream:
    curl -N -X POST http://localhost:8001/stream \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer <token>" \
      -d '{"query": "What is the attendance policy?", "tenant_id": "anna_university"}'
"""

import os
import json
import time
import asyncio
from typing import AsyncGenerator

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from domain_classifier import classify_domain
from role_access import get_current_user, check_permission
from latency_tracker import LatencyTracker, get_latency_summary   # ⏱️ NEW

# ─── CONFIG ───────────────────────────────────────────────────────────────────

CHROMA_DIR       = "./chroma_db"
TOP_K            = 3       # number of chunks to retrieve
CHUNK_DELAY      = 0.03    # seconds between streamed words (tweak for speed)

# Confidence thresholds
HIGH_CONFIDENCE   = 0.75
MEDIUM_CONFIDENCE = 0.45

# Disclaimer templates
DISCLAIMERS = {
    "education": "⚠️ Please verify this information with your official college portal or academic advisor.",
    "healthcare": "⚠️ This is for informational purposes only. Always consult a qualified medical professional.",
    "general":    "⚠️ This answer is based on available documents. Please verify with official sources.",
}

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Secure AI Agent — Streaming RAG",
    description="SSE Streaming + Confidence Score + Multi-tenant RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CHROMA CLIENT ────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# ─── REQUEST MODEL ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    tenant_id: str


# ─── CONFIDENCE CALCULATOR ────────────────────────────────────────────────────

def calculate_confidence(distances: list[float]) -> dict:
    """
    Convert ChromaDB cosine distances → confidence score.

    ChromaDB cosine distance: 0 = perfect match, 2 = opposite
    We convert to similarity: 1 - (distance / 2) → 0 to 1

    Args:
        distances: list of cosine distances from ChromaDB

    Returns:
        dict with score (0-100), label, and color
    """
    if not distances:
        return {"score": 0, "label": "Low", "color": "red"}

    similarities = [1 - (d / 2) for d in distances]
    avg_similarity = sum(similarities) / len(similarities)
    score = round(avg_similarity * 100, 1)

    if avg_similarity >= HIGH_CONFIDENCE:
        label = "High"
        color = "green"
    elif avg_similarity >= MEDIUM_CONFIDENCE:
        label = "Medium"
        color = "orange"
    else:
        label = "Low"
        color = "red"

    return {"score": score, "label": label, "color": color}


# ─── RETRIEVAL ────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, collection_name: str, top_k: int = TOP_K) -> dict:
    """
    Retrieve top-k relevant chunks from ChromaDB.
    """
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception:
        return {
            "chunks": [],
            "sources": [],
            "distances": [],
            "confidence": {"score": 0, "label": "Low", "color": "red"},
            "error": f"Collection '{collection_name}' not found"
        }

    count = collection.count()
    if count == 0:
        return {
            "chunks": [],
            "sources": [],
            "distances": [],
            "confidence": {"score": 0, "label": "Low", "color": "red"},
            "error": "Collection is empty"
        }

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"]
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    sources   = list(dict.fromkeys([m.get("source_file", "unknown") for m in metadatas]))

    confidence = calculate_confidence(distances)

    return {
        "chunks": chunks,
        "sources": sources,
        "distances": distances,
        "confidence": confidence,
        "error": None
    }


# ─── ANSWER GENERATOR ─────────────────────────────────────────────────────────

def build_answer(query: str, chunks: list[str], domain: str, confidence: dict) -> str:
    """
    Build answer from retrieved chunks.
    (Swap with LLM call when credits available)
    """
    if not chunks:
        return "I could not find relevant information in the documents for your query."

    top_chunk = chunks[0]
    words = top_chunk.split()
    answer = " ".join(words[:150]) + ("..." if len(words) > 150 else "")

    needs_disclaimer = (
        confidence["label"] == "Low" or
        domain in ["healthcare", "education"]
    )

    if needs_disclaimer:
        disclaimer = DISCLAIMERS.get(domain, DISCLAIMERS["general"])
        answer = f"{answer}\n\n{disclaimer}"

    return answer


# ─── SSE STREAM GENERATOR ─────────────────────────────────────────────────────

async def stream_response(
    query: str,
    tenant_id: str,
    user: dict
) -> AsyncGenerator[str, None]:
    """
    Core SSE generator — streams response word by word.
    """

    # ⏱️ Init latency tracker
    tracker = LatencyTracker(
        query=query,
        user_id=user.get("user_id", "unknown"),
        role=user.get("role", "unknown"),
        domain=user.get("domain", "unknown"),
    )

    # Step 1: Classify domain
    domain_result = classify_domain(query)
    domain = domain_result["domain"]
    if domain == "general":
        domain = "education"

    # Step 2: Collection name
    collection_name = f"{domain}_{tenant_id}"

    # Step 3: Retrieve ⏱️
    tracker.start("retrieval")
    retrieval = retrieve_chunks(query, collection_name)
    tracker.stop("retrieval")

    if retrieval["error"]:
        tracker.finish()
        tracker.log()
        yield f"data: {json.dumps({'type': 'error', 'message': retrieval['error']})}\n\n"
        return

    confidence = retrieval["confidence"]
    sources    = retrieval["sources"]
    chunks     = retrieval["chunks"]

    # Step 4: Reranking ⏱️
    tracker.start("reranking")
    await asyncio.sleep(0)
    tracker.stop("reranking")

    # Step 5: Metadata event
    metadata_event = {
        "type":       "metadata",
        "confidence": {
            "score": confidence["score"],
            "label": confidence["label"],
            "color": confidence["color"],
        },
        "sources":    sources,
        "domain":     domain,
        "collection": collection_name,
        "tenant_id":  tenant_id,
        "role":       user.get("role", "unknown"),
    }
    yield f"data: {json.dumps(metadata_event)}\n\n"
    await asyncio.sleep(0)

    # Step 6: Stream tokens ⏱️
    answer = build_answer(query, chunks, domain, confidence)

    tracker.start("generation")
    words = answer.split(" ")
    first_token_marked = False

    for word in words:
        token_event = {"type": "token", "token": word + " "}
        yield f"data: {json.dumps(token_event)}\n\n"

        # ⏱️ TTFT on first token
        if not first_token_marked:
            tracker.mark_first_token()
            first_token_marked = True

        await asyncio.sleep(CHUNK_DELAY)

    tracker.stop("generation")

    # Step 7: Done + latency report ⏱️
    tracker.finish()
    report = tracker.log()

    done_event = {
        "type":             "done",
        "message":          "Stream complete",
        "latency":          report["latency_ms"],
        "performance_grade": report["performance_grade"],
    }
    yield f"data: {json.dumps(done_event)}\n\n"


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.post("/stream")
async def stream_query(
    request: QueryRequest,
    authorization: str = Header(...)
):
    """SSE streaming endpoint. Requires: Authorization: Bearer <jwt_token>"""
    try:
        user = get_current_user(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        check_permission(user, action="query", target_tenant_id=request.tenant_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    return StreamingResponse(
        stream_response(request.query, request.tenant_id, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/health")
def health():
    """Quick health check."""
    collections = [c.name for c in chroma_client.list_collections()]
    return {
        "status":           "ok",
        "collections":      collections,
        "collection_count": len(collections),
    }


@app.get("/confidence-test")
def confidence_test():
    """Test confidence calculation with sample distances."""
    tests = [
        [0.1, 0.15, 0.2],
        [0.5, 0.6, 0.7],
        [1.2, 1.4, 1.6],
    ]
    results = []
    for distances in tests:
        conf = calculate_confidence(distances)
        results.append({"distances": distances, **conf})
    return {"confidence_tests": results}


# ⏱️ NEW — Latency stats endpoint
@app.get("/latency-stats")
def latency_stats():
    """Return aggregate latency statistics from metrics/latency.jsonl"""
    return get_latency_summary()


# ─── TEST ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("CONFIDENCE SCORE — QUICK TEST")
    print("=" * 60)

    test_cases = [
        ([0.1, 0.15, 0.2],  "Should be HIGH"),
        ([0.5, 0.6, 0.65],  "Should be MEDIUM"),
        ([1.2, 1.4, 1.5],   "Should be LOW"),
        ([],                 "Should be LOW (empty)"),
    ]

    for distances, expected in test_cases:
        result = calculate_confidence(distances)
        print(f"\n  {expected}")
        print(f"  Distances : {distances}")
        print(f"  Score     : {result['score']}%")
        print(f"  Label     : {result['label']}")

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)

    test_queries = [
        ("What is the attendance policy?", "education_anna_university"),
        ("What are the exam dates?",       "education_anna_university"),
        ("What is the fee structure?",     "education_anna_university"),
    ]

    for query, collection in test_queries:
        result = retrieve_chunks(query, collection)
        conf   = result["confidence"]
        print(f"\n  Query     : {query}")
        print(f"  Collection: {collection}")
        if result["error"]:
            print(f"  Error     : {result['error']}")
        else:
            print(f"  Chunks    : {len(result['chunks'])} retrieved")
            print(f"  Sources   : {result['sources']}")
            print(f"  Confidence: {conf['score']}% ({conf['label']})")

    print("\n✅ Tests complete! Starting server...")
    print("📡 API docs : http://localhost:8001/docs")
    print("🔍 Health   : http://localhost:8001/health")
    print("⏱️  Latency  : http://localhost:8001/latency-stats")
    uvicorn.run(app, host="0.0.0.0", port=8001)