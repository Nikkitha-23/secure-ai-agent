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
    Improved planner with intelligent routing.
    Returns: source strategy, query rewriting decision, depth, and route info
    """
    # 🧭 Smart routing
    route_info = route_sources(query)
    
    prompt = f"""You are an AI planner. Given a user query, return a JSON plan.
Recommended source routing: {route_info['reasoning']}

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
            plan["route_info"] = route_info
            logging.info(f"📋 Plan: {plan}")
            logging.info(f"🧭 Route: {route_info['reasoning']} (confidence: {route_info['confidence']})")
            return plan
    except Exception as e:
        logging.warning(f"⚠️ Planner failed: {e} → using smart defaults")

    # Fallback to smart routing
    return {
        "source": route_info["primary"],
        "needs_rewrite": True,
        "reasoning_depth": "simple",
        "route_info": route_info
    }

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

# ── Smart Source Router ──────────────────────────────────────────────────────
def route_sources(query: str) -> dict:
    """
    Intelligent source routing based on query characteristics.
    Returns: {primary: str, fallback: str, confidence: float, reasoning: str}
    """
    query_lower = query.lower()
    
    # Freshness indicators (recent events, current info)
    freshness_keywords = ["latest", "2025", "2026", "recent", "today", "this week", 
                         "current", "now", "news", "today's", "breaking", "just", "new"]
    needs_fresh = any(kw in query_lower for kw in freshness_keywords)
    
    # Academic/Technical indicators (likely in PDFs)
    academic_keywords = ["algorithm", "architecture", "framework", "protocol", "theorem",
                        "research", "study", "analysis", "design", "implementation",
                        "academic", "textbook", "theory", "concept"]
    is_academic = any(kw in query_lower for kw in academic_keywords)
    
    # Practical/How-to indicators (could be either)
    practical_keywords = ["how to", "tutorial", "guide", "step", "setup", "install",
                         "configure", "build", "create", "make"]
    is_practical = any(kw in query_lower for kw in practical_keywords)
    
    # Decision logic
    if needs_fresh:
        # Recent events need web search
        primary, fallback = "web", "pdf"
        confidence = 0.9
        reasoning = "Query requires current information"
    elif is_academic and not needs_fresh:
        # Academic topics preferably from PDFs
        primary, fallback = "pdf", "web"
        confidence = 0.85
        reasoning = "Academic/technical query - PDF likely has authoritative sources"
    elif is_practical:
        # Practical info from both, start with web
        primary, fallback = "both", "both"
        confidence = 0.75
        reasoning = "Practical query - checking both PDF guides and web resources"
    else:
        # Default: start with PDF, fallback to web
        primary, fallback = "pdf", "web"
        confidence = 0.6
        reasoning = "General query - starting with PDF, fallback to web"
    
    return {
        "primary": primary,
        "fallback": fallback,
        "confidence": confidence,
        "reasoning": reasoning,
        "needs_fresh": needs_fresh,
        "is_academic": is_academic
    }

# ── Complexity Detector ──────────────────────────────────────────────────────
def detect_complexity(query: str) -> dict:
    """
    Analyzes query complexity.
    Returns: {level: "simple|moderate|complex", criteria: dict}
    """
    query_lower = query.lower()
    word_count = len(query.split())
    
    # Complexity indicators
    complex_keywords = ["compare", "contrast", "analyze", "summarize", "discuss", "explain how", 
                       "explain why", "what if", "how would", "advantages", "disadvantages", 
                       "pros and cons", "relationship", "impact", "effect"]
    
    technical_keywords = ["algorithm", "architecture", "framework", "protocol", "implement"]
    multi_part_indicators = [" and ", " or ", ",", ";"]
    
    complexity_score = 0
    
    # Check for complex keywords
    if any(kw in query_lower for kw in complex_keywords):
        complexity_score += 2
    
    # Check for technical terms
    if any(kw in query_lower for kw in technical_keywords):
        complexity_score += 1
    
    # Check for multi-part questions
    multi_part_count = sum(query_lower.count(sep) for sep in multi_part_indicators)
    complexity_score += min(multi_part_count, 2)
    
    # Word count indicator
    if word_count > 20:
        complexity_score += 1
    elif word_count < 5:
        complexity_score -= 1
    
    # Determine level and set criteria
    if complexity_score >= 4:
        level = "complex"
        criteria = {
            "min_words": 10,
            "description": "Complex query - requires detailed, multi-faceted answer",
            "strict": False
        }
    elif complexity_score >= 2:
        level = "moderate"
        criteria = {
            "min_words": 7,
            "description": "Moderate query - needs balanced answer",
            "strict": False
        }
    else:
        level = "simple"
        criteria = {
            "min_words": 5,
            "description": "Simple query - concise answer acceptable",
            "strict": True
        }
    
    logging.info(f"📊 Query Complexity: {level} (score: {complexity_score}) | Criteria: {criteria['min_words']} words min")
    
    return {
        "level": level,
        "score": complexity_score,
        "criteria": criteria
    }

# ── Step 3: Reflector ─────────────────────────────────────────────────────────
def reflect(answer: str, query: str) -> dict:
    """
    Agent checks its own answer quality with complexity-aware criteria.
    Returns: {good: True/False, reason: str, retry_source: str, complexity: str}
    """
    # 📊 Detect query complexity
    complexity_info = detect_complexity(query)
    complexity_level = complexity_info["level"]
    min_words = complexity_info["criteria"]["min_words"]
    
    prompt = f"""You are a quality checker. Given a question and answer, evaluate if the answer is good.

Query Complexity Level: {complexity_level}
Minimum acceptable length: {min_words} words

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
- It is too short (less than {min_words} words)
- It doesn't answer the question
- For complex queries: missing important details or nuance"""

    try:
        raw = call_llm(prompt)
        import json, re
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            reflection = json.loads(match.group())
            reflection["complexity"] = complexity_level
            logging.info(f"🔍 Reflection [{complexity_level}]: {reflection}")
            return reflection
    except Exception as e:
        logging.warning(f"⚠️ Reflection failed: {e}")

    return {"good": True, "reason": "default pass", "retry_source": "web", "complexity": complexity_level}

# ── MAIN AGENT LOOP ───────────────────────────────────────────────────────────
def run_agent(query: str, session_id: str = "default") -> dict:
    """
    Full agentic loop with adaptive retry strategies:
    Plan → Execute → Answer → Reflect → Adaptive Retry → Memory → Return
    """
    logging.info(f"\n{'='*50}")
    logging.info(f"🤖 Agent started for: {query}")

    # Memory recall
    past_context = memory.recall(query, session_id)

    # STEP 1: Plan
    plan = planner(query)
    source = plan.get("source", "pdf")
    route_info = plan.get("route_info", {})
    routing_confidence = route_info.get("confidence", 0.5)
    
    logging.info(f"📊 Routing confidence: {routing_confidence}")
    
    # Determine retry threshold based on routing confidence
    # Low confidence = lower threshold = more aggressive retries
    if routing_confidence >= 0.85:
        max_retries_allowed = 1  # High confidence: accept first good answer
        retry_threshold = 0.95
    elif routing_confidence >= 0.75:
        max_retries_allowed = 2  # Moderate confidence: 2 chances
        retry_threshold = 0.85
    else:
        max_retries_allowed = MAX_RETRIES  # Low confidence: full retry limit
        retry_threshold = 0.7
    
    logging.info(f"🎯 Retry strategy: max {max_retries_allowed} attempts, quality threshold {retry_threshold}")

    # STEP 2: Rewrite if needed
    if plan.get("needs_rewrite", True):
        query = rewrite_query(query)
        logging.info(f"✏️ Rewritten: {query}")

    attempt = 0
    answer = ""
    reflection = {"good": False}

    # STEP 3: Agentic Loop — retry until good answer or max retries
    while attempt < max_retries_allowed and not reflection.get("good", False):
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

        # If bad — adaptive retry strategy
        if not reflection.get("good", False):
            retry_source = reflection.get("retry_source", "both")
            
            # Adaptive strategy based on routing confidence and attempts
            if routing_confidence < 0.7:
                # Low confidence route = switch more aggressively
                if attempt == 1:
                    source = "both"  # First retry: expand to both sources
                    logging.warning(f"🔄 Low confidence route — expanding to both sources")
                else:
                    source = retry_source or "both"  # Later retries: defer to reflection
                    logging.warning(f"🔄 Attempt {attempt}: using reflection suggestion")
            else:
                # High confidence route = defer to reflection judgment
                source = retry_source or "both"
                logging.warning(f"🔄 Attempt {attempt}: {reflection.get('reason', 'retrying')}")
            
            # Log strategy
            logging.info(f"📋 Retry strategy: confidence={routing_confidence:.2f}, source={source}")

    # STEP 5: Memory save
    memory.save(query, answer, session_id)
    logging.info(f"✅ Agent done after {attempt} attempt(s) | Quality: {'✅' if reflection.get('good') else '❌'}")

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