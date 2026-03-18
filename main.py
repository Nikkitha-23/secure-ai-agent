"""
main.py — FastAPI with Full Agentic RAG Pipeline + Memory + Monitor + Cache
---------------------------------------------------------------------------
Flow:
1. Security Check
2. Cache Check         ← NEW (instant return if cached)
3. Memory Recall
4. Query Rewriting
5. Smart Router        → pdf / web / both
6. Retrieve            → ChromaDB and/or Tavily
7. Re-Ranker           → top 3 chunks
8. LLM (Groq)          → final answer
9. Memory Save
10. Cache Save         ← NEW
11. Monitor Log
"""

from fastapi import FastAPI
from security.filter import check_input
from pydantic import BaseModel
from rag.retrieve import get_retriever
from rag.reranker import rerank_documents
from rag.router import decide_source
from rag.web_search import web_search
from rag.query_rewriter import rewrite_query
from rag.memory import get_memory
from rag.monitor import monitor
from rag.cache import cache              # ← NEW
from functools import lru_cache
from dotenv import load_dotenv
import logging
import uuid
import os

load_dotenv()

from groq import Groq

logging.basicConfig(level=logging.INFO)

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@lru_cache(maxsize=1)
def get_cached_retriever():
    return get_retriever()

retriever = get_cached_retriever()
memory = get_memory()

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question
    session_id = request.session_id
    query_id = str(uuid.uuid4())
    monitor.start(query_id)
    logging.info(f"📩 User query: {query}")

    # 🔐 STEP 1: Security Check
    result = check_input(query)
    if result["status"] == "block":
        monitor.end(query_id, question=query, search_type="blocked", status="blocked")
        logging.warning(f"🚫 Blocked query: {query}")
        return {"answer": "⚠️ Your input contains unsafe instructions and was blocked.", "sources": [], "search_type": "blocked"}

    clean_query = result["clean_input"]

    # ⚡ STEP 2: Cache Check
    cached_response = cache.get(clean_query)
    if cached_response:
        monitor.end(query_id, question=clean_query, search_type="cache", status="cache_hit")
        logging.info(f"⚡ Cache HIT — returning instantly!")
        return cached_response

    try:
        # 🧠 STEP 3: Memory Recall
        past_context = memory.recall(clean_query, session_id)
        if past_context:
            logging.info(f"🧠 Memory recalled for session: {session_id}")

        # ✏️ STEP 4: Query Rewriting
        rewritten_query = rewrite_query(clean_query)

        # 🧭 STEP 5: Smart Router
        source = decide_source(rewritten_query)
        all_docs = []

        # 📚 STEP 6a: PDF Retrieval
        if source in ["pdf", "both"]:
            pdf_docs = retriever.invoke(rewritten_query)
            logging.info(f"📄 PDF chunks retrieved: {len(pdf_docs)}")
            all_docs.extend(pdf_docs)

        # 🌐 STEP 6b: Web Search
        if source in ["web", "both"]:
            web_docs = web_search(rewritten_query, max_results=3)
            logging.info(f"🌐 Web results retrieved: {len(web_docs)}")
            all_docs.extend(web_docs)

        if not all_docs:
            monitor.end(query_id, question=clean_query, search_type=source, status="no_results")
            return {
                "answer": "❌ No relevant information found. Please try rephrasing your question.",
                "sources": [],
                "search_type": source,
                "rewritten_query": rewritten_query
            }

        # 🏆 STEP 7: Re-Rank
        reranked_docs = rerank_documents(rewritten_query, all_docs, top_n=3)
        logging.info(f"✅ Re-ranked: top {len(reranked_docs)} chunks selected")

        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        memory_section = f"\n{past_context}\n" if past_context else ""

        prompt = f"""
You are a document question-answering system.

STRICT RULES:
- Use ONLY the EXACT information from the context below.
- Copy key facts DIRECTLY from context — do not paraphrase.
- Do NOT use outside knowledge under any circumstances.
- Do NOT add any information not present in context.
- If the answer is in past conversations, you may use it.
- If answer not in context, say ONLY: "Not found in documents."
{memory_section}
Context:
{context}

Question:
{clean_query}

Answer (use only context facts):"""

        # 🚀 STEP 8: Groq LLM Call
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a secure and helpful college AI assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content
        logging.info("✅ LLM response received")

        # 💾 STEP 9: Memory Save
        memory.save(clean_query, answer, session_id)

        # 📊 STEP 10: Monitor Log
        monitor.end(
            query_id,
            question=clean_query,
            search_type=source,
            input_tokens=len(prompt.split()),
            output_tokens=len(answer.split()),
            status="success"
        )

        sources = list(set([
            doc.metadata.get("source", "unknown") for doc in reranked_docs
        ]))

        response_data = {
            "answer": answer,
            "sources": sources,
            "context": [doc.page_content for doc in reranked_docs],
            "search_type": source,
            "rewritten_query": rewritten_query
        }

        # ⚡ STEP 11: Cache Save
        cache.set(clean_query, response_data)

        return response_data

    except ConnectionError as e:
        monitor.end(query_id, question=clean_query, status="error")
        logging.error(f"Connection error: {e}")
        return {"answer": "⚠️ Could not connect. Please try again.", "sources": [], "search_type": "error"}

    except Exception as e:
        monitor.end(query_id, question=clean_query, status="error")
        logging.error(f"Unexpected error: {e}")
        return {"answer": "⚠️ Something went wrong. Please try again.", "sources": [], "search_type": "error"}


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "✅ EduBot API is running"}

# ── Memory endpoints ───────────────────────────────────────────────────────────
@app.delete("/memory/clear")
def clear_memory(session_id: str = "default"):
    memory.clear(session_id)
    return {"status": f"✅ Memory cleared for session: {session_id}"}

@app.get("/memory/recent")
def recent_memory(session_id: str = "default"):
    recent = memory.get_recent(session_id)
    return {"session_id": session_id, "conversations": recent}

# ── Monitor endpoints ──────────────────────────────────────────────────────────
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

# ── Cache endpoints ────────────────────────────────────────────────────────────
@app.get("/cache/stats")
def cache_stats():
    return cache.stats()

@app.delete("/cache/clear")
def cache_clear():
    cache.clear()
    return {"status": "✅ Cache cleared"}