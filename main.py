"""
main.py — FastAPI with Full Agentic RAG Pipeline
-------------------------------------------------
Flow:
1. Security Check
2. Query Rewriting     ← NEW
3. Smart Router        → pdf / web / both
4. Retrieve            → ChromaDB and/or Tavily
5. Re-Ranker           → top 3 chunks
6. LLM (Groq)          → final answer
"""

from fastapi import FastAPI
from security.filter import check_input
from pydantic import BaseModel
from rag.retrieve import get_retriever
from rag.reranker import rerank_documents
from rag.router import decide_source
from rag.web_search import web_search
from rag.query_rewriter import rewrite_query          # ← NEW
from functools import lru_cache
from dotenv import load_dotenv
import logging
import os

load_dotenv()

from groq import Groq

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)

# ── App & Client ───────────────────────────────────────────────────────────────
app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Cached Retriever ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_cached_retriever():
    return get_retriever()

retriever = get_cached_retriever()

# ── Request model ──────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

# ── /ask endpoint ──────────────────────────────────────────────────────────────
@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.question
    logging.info(f"📩 User query: {query}")

    # 🔐 STEP 1: Security Check
    result = check_input(query)
    if result["status"] == "block":
        logging.warning(f"🚫 Blocked query: {query}")
        return {"answer": "⚠️ Your input contains unsafe instructions and was blocked.", "sources": [], "search_type": "blocked"}

    clean_query = result["clean_input"]

    try:

        # ✏️ STEP 2: Query Rewriting — make the query better before searching
        rewritten_query = rewrite_query(clean_query)

        # 🧭 STEP 3: Smart Router — pdf / web / both
        # Router uses rewritten query for better decision too
        source = decide_source(rewritten_query)
        all_docs = []

        # 📚 STEP 4a: PDF Retrieval (ChromaDB)
        if source in ["pdf", "both"]:
            pdf_docs = retriever.invoke(rewritten_query)
            logging.info(f"📄 PDF chunks retrieved: {len(pdf_docs)}")
            all_docs.extend(pdf_docs)

        # 🌐 STEP 4b: Web Search (Tavily)
        if source in ["web", "both"]:
            web_docs = web_search(rewritten_query, max_results=3)
            logging.info(f"🌐 Web results retrieved: {len(web_docs)}")
            all_docs.extend(web_docs)

        # ── No results ─────────────────────────────────────────────────────────
        if not all_docs:
            return {
                "answer": "❌ No relevant information found. Please try rephrasing your question.",
                "sources": [],
                "search_type": source,
                "rewritten_query": rewritten_query
            }

        # 🏆 STEP 5: Re-Rank → top 3 most relevant chunks
        reranked_docs = rerank_documents(rewritten_query, all_docs, top_n=3)
        logging.info(f"✅ Re-ranked: top {len(reranked_docs)} chunks selected")

        context = "\n\n".join([doc.page_content for doc in reranked_docs])

        # 🧠 STEP 6: Prompt Building
        source_instruction = {
            "pdf":  "Use ONLY the college document context provided.",
            "web":  "Use ONLY the web search results provided.",
            "both": "Use the college documents AND web search results provided."
        }.get(source, "Use ONLY the context provided.")

        prompt = f"""
You are a document question-answering system.

STRICT RULES:
- Use ONLY the EXACT information from the context below.
- Copy key facts DIRECTLY from context — do not paraphrase.
- Do NOT use outside knowledge under any circumstances.
- Do NOT add any information not present in context.
- If answer not in context, say ONLY: "Not found in documents."

Context:
{context}

Question:
{clean_query}

Answer (use only context facts):"""

        # 🚀 STEP 7: Groq LLM Call
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

        # 📎 STEP 8: Sources
        sources = list(set([
            doc.metadata.get("source", "unknown") for doc in reranked_docs
        ]))

        return {
            "answer": answer,
            "sources": sources,
            "context": [doc.page_content for doc in reranked_docs],  # ← ADD THIS
            "search_type": source,
            "rewritten_query": rewritten_query    # useful for UI/debugging
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
