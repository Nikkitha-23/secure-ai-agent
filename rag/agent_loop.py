"""
agent_loop.py — Real Agentic Loop
----------------------------------
Real Agent Flow:
User → Planner → Tool Decision → Execute → Reflect → Memory → Response
       ↑______________retry if bad answer___________________↑
"""

from rag.router import decide_source
from rag.retrieve import get_retriever
from rag.reranker import rerank_documents
from rag.web_search import web_search
from rag.query_rewriter import rewrite_query
from rag.memory import get_memory_manager
from dotenv import load_dotenv
from groq import Groq
import logging
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
memory = get_memory_manager()
retriever = get_retriever()

MAX_RETRIES = 3  # Agent max retry count

# ── LLM Call ──────────────────────────────────────────────────────────────────
def call_llm(prompt: str) -> str:
    response = llm.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a secure and helpful college AI assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ── Step 1: Planner ───────────────────────────────────────────────────────────
def planner(query: str) -> dict:
    """
    Decides:
    - source: pdf / web / both
    - needs_rewrite: True/False
    - reasoning_depth: simple / deep
    """
    prompt = f"""You are an AI planner. Given a user query, return a JSON plan.

Query: {query}

Return ONLY this JSON format, nothing else:
{{
  "source": "pdf or web or both",
  "needs_rewrite": true or false,
  "reasoning_depth": "simple or deep"
}}"""

    try:
        raw = call_llm(prompt)
        import json, re
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            plan = json.loads(match.group())
            logging.info(f"📋 Plan: {plan}")
            return plan
    except Exception as e:
        logging.warning(f"⚠️ Planner failed: {e} → using defaults")

    return {"source": "pdf", "needs_rewrite": True, "reasoning_depth": "simple"}

# ── Step 2: Tool Executor ─────────────────────────────────────────────────────
def execute_tools(query: str, source: str) -> list:
    """Runs retrieval tools based on planner decision"""
    all_docs = []

    if source in ["pdf", "both"]:
        pdf_docs = retriever.invoke(query)
        logging.info(f"📄 PDF docs: {len(pdf_docs)}")
        all_docs.extend(pdf_docs)

    if source in ["web", "both"]:
        web_docs = web_search(query, max_results=3)
        logging.info(f"🌐 Web docs: {len(web_docs)}")
        all_docs.extend(web_docs)

    return all_docs

# ── Step 3: Reflector ─────────────────────────────────────────────────────────
def reflect(answer: str, query: str) -> dict:
    """
    Agent checks its own answer quality.
    Returns: {good: True/False, reason: str, retry_source: str}
    """
    prompt = f"""You are a quality checker. Given a question and answer, evaluate if the answer is good.

Question: {query}
Answer: {answer}

Return ONLY this JSON:
{{
  "good": true or false,
  "reason": "why good or bad in one sentence",
  "retry_source": "web or pdf or both"
}}

Answer is BAD if:
- It says "not found" or "I don't know"
- It is too short (less than 5 words)
- It doesn't answer the question"""

    try:
        raw = call_llm(prompt)
        import json, re
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            reflection = json.loads(match.group())
            logging.info(f"🔍 Reflection: {reflection}")
            return reflection
    except Exception as e:
        logging.warning(f"⚠️ Reflection failed: {e}")

    return {"good": True, "reason": "default pass", "retry_source": "web"}

# ── MAIN AGENT LOOP ───────────────────────────────────────────────────────────
def run_agent(query: str, session_id: str = "default") -> dict:
    """
    Full agentic loop:
    Plan → Execute → Answer → Reflect → Retry if bad → Memory → Return
    """
    logging.info(f"\n{'='*50}")
    logging.info(f"🤖 Agent started for: {query}")

    # Memory recall
    past_context = memory.recall(query, session_id)

    # STEP 1: Plan
    plan = planner(query)
    source = plan.get("source", "pdf")

    # STEP 2: Rewrite if needed
    if plan.get("needs_rewrite", True):
        query = rewrite_query(query)
        logging.info(f"✏️ Rewritten: {query}")

    attempt = 0
    answer = ""
    reflection = {"good": False}

    # STEP 3: Agentic Loop — retry until good answer or max retries
    while attempt < MAX_RETRIES and not reflection.get("good", False):
        attempt += 1
        logging.info(f"🔄 Attempt {attempt}/{MAX_RETRIES} | Source: {source}")

        # Execute tools
        docs = execute_tools(query, source)

        if not docs:
            logging.warning(f"⚠️ No docs found — switching to both")
            source = "both"
            continue

        # Rerank
        reranked = rerank_documents(query, docs, top_n=3)
        context = "\n\n".join([doc.page_content for doc in reranked])
        memory_section = f"\nPast context:\n{past_context}\n" if past_context else ""

        # Generate answer
        prompt = f"""You are a helpful college AI assistant.
{memory_section}
Context:
{context}

Question: {query}

Answer using ONLY the context. If not found, say "Not found in documents."
Answer:"""

        answer = call_llm(prompt)
        logging.info(f"💬 Answer attempt {attempt}: {answer[:100]}...")

        # STEP 4: Reflect
        reflection = reflect(answer, query)

        # If bad — switch source and retry
        if not reflection.get("good", False):
            logging.warning(f"🔄 Bad answer — retrying with: {reflection.get('retry_source', 'both')}")
            source = reflection.get("retry_source", "both")

    # STEP 5: Memory save
    memory.save(query, answer, session_id)
    logging.info(f"✅ Agent done after {attempt} attempt(s)")

    sources = list(set([
        doc.metadata.get("source", "unknown") for doc in reranked
    ])) if 'reranked' in locals() else []

    return {
        "answer": answer,
        "sources": sources,
        "attempts": attempt,
        "plan": plan,
        "reflection": reflection,
        "search_type": source
    }