"""
main.py — FastAPI with Full Agentic RAG Pipeline + Memory
----------------------------------------------------------
Flow:
1. Security Check
2. Memory Recall       ← NEW (past conversations)
3. Query Rewriting
4. Smart Router        → pdf / web / both
5. Retrieve            → ChromaDB and/or Tavily
6. Re-Ranker           → top 3 chunks
7. LLM (Groq)          → final answer
8. Memory Save         ← NEW (save this conversation)
"""

from fastapi import FastAPI
from security.filter import check_input
from pydantic import BaseModel
from rag.retrieve import get_retriever
from rag.reranker import rerank_documents
from rag.router import decide_source
from rag.web_search import web_search
from rag.query_rewriter import rewrite_query
from rag.memory import get_memory          # ← NEW
from functools import lru_cache
from dotenv import load_dotenv
import logging
import os

load_dotenv()

from groq import Groq

logging.basicConfig(level=logging.INFO)

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Cached Retriever ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_cached_retriever():
    return get_retriever()

retriever = get_cached_retriever()
memory = get_memory()                      # ← NEW

# ── Request model ──────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"            # ← NEW (optional, default session)

# ── /ask endpoint ──────────────────────────────────────────────────────────────
@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question
    session_id = request.session_id        # ← NEW
    logging.info(f"📩 User query: {query}")

    # 🔐 STEP 1: Security Check
    result = check_input(query)
    if result["status"] == "block":
        logging.warning(f"🚫 Blocked query: {query}")
        return {"answer": "⚠️ Your input contains unsafe instructions and was blocked.", "sources": [], "search_type": "blocked"}

    clean_query = result["clean_input"]

    try:
        # 🧠 STEP 2: Memory Recall — past conversations தேடு
        past_context = memory.recall(clean_query, session_id)
        if past_context:
            logging.info(f"🧠 Memory recalled for session: {session_id}")

        # ✏️ STEP 3: Query Rewriting
        rewritten_query = rewrite_query(clean_query)

        # 🧭 STEP 4: Smart Router
        source = decide_source(rewritten_query)
        all_docs = []

        # 📚 STEP 5a: PDF Retrieval (ChromaDB)
        if source in ["pdf", "both"]:
            pdf_docs = retriever.invoke(rewritten_query)
            logging.info(f"📄 PDF chunks retrieved: {len(pdf_docs)}")
            all_docs.extend(pdf_docs)

        # 🌐 STEP 5b: Web Search (Tavily)
        if source in ["web", "both"]:
            web_docs = web_search(rewritten_query, max_results=3)
            logging.info(f"🌐 Web results retrieved: {len(web_docs)}")
            all_docs.extend(web_docs)

        if not all_docs:
            return {
                "answer": "❌ No relevant information found. Please try rephrasing your question.",
                "sources": [],
                "search_type": source,
                "rewritten_query": rewritten_query
            }

        # 🏆 STEP 6: Re-Rank
        reranked_docs = rerank_documents(rewritten_query, all_docs, top_n=3)
        logging.info(f"✅ Re-ranked: top {len(reranked_docs)} chunks selected")

        context = "\n\n".join([doc.page_content for doc in reranked_docs])

        # 🧠 STEP 7: Prompt — memory context include பண்ணு
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

        # 💾 STEP 9: Memory Save — இந்த conversation save பண்ணு
        memory.save(clean_query, answer, session_id)  # ← NEW
        logging.info(f"💾 Conversation saved to memory")

        sources = list(set([
            doc.metadata.get("source", "unknown") for doc in reranked_docs
        ]))

        return {
            "answer": answer,
            "sources": sources,
            "context": [doc.page_content for doc in reranked_docs],
            "search_type": source,
            "rewritten_query": rewritten_query
        }

    except ConnectionError as e:
        logging.error(f"Connection error: {e}")
        return {"answer": "⚠️ Could not connect. Please try again.", "sources": [], "search_type": "error"}

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"answer": "⚠️ Something went wrong. Please try again.", "sources": [], "search_type": "error"}


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "✅ EduBot API is running"}


# ── Clear memory endpoint ──────────────────────────────────────────────────────
@app.delete("/memory/clear")
def clear_memory(session_id: str = "default"):
    memory.clear(session_id)
    return {"status": f"✅ Memory cleared for session: {session_id}"}


# ── View recent memory endpoint ────────────────────────────────────────────────
@app.get("/memory/recent")
def recent_memory(session_id: str = "default"):
    recent = memory.get_recent(session_id)
    return {"session_id": session_id, "conversations": recent}
