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
    
    # Check for sequential/chained queries (step-by-step reasoning)
    sequential_keywords = [" then ", " and then ", " after ", " next "]
    if any(kw in query_lower for kw in sequential_keywords):
        complexity_score += 1

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
        "criteria": criteria,
        "min_words": criteria["min_words"]
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

# ── Step 3.1: Query Decomposition ────────────────────────────────────────────
def decompose_query(query: str) -> dict:
    """
    Analyzes if query should be decomposed into sub-tasks.
    Returns: {should_decompose: bool, subtasks: list, reasoning: str}
    """
    complexity_info = detect_complexity(query)
    complexity_level = complexity_info["level"]
    
    # Only decompose complex queries — UNLESS it has freshness + explanation pattern
    query_lower = query.lower()
    freshness_words = ["latest", "recent", "new", "current"]
    explain_words = ["explain", "describe", "understand", "how", "what's"]
    has_and = " and " in query_lower
    has_freshness_explain = any(f in query_lower for f in freshness_words) and has_and and any(e in query_lower for e in explain_words)
    
    if complexity_level == "simple" and not has_freshness_explain:
        return {
            "should_decompose": False,
            "subtasks": [],
            "reasoning": f"Query is {complexity_level} - single-step execution sufficient"
        }
    
    if complexity_level == "moderate" and not has_freshness_explain:
        return {
            "should_decompose": False,
            "subtasks": [],
            "reasoning": f"Query is {complexity_level} - single-step execution sufficient"
        }
    
    # Multi-part question patterns: AND (connect), multiple commas, OR (alternatives)
    has_and = " and " in query_lower or ", " in query_lower
    has_or = " or " in query_lower
    
    # Check for comparison/contrast words + multi-part
    comparison_words = ["compare", "contrast", "difference", "vs", "versus"]
    has_comparison = any(word in query_lower for word in comparison_words)
    
    # Check for freshness indicators combined with explanation
    freshness_words = ["latest", "recent", "new", "current"]
    explain_words = ["explain", "describe", "understand", "how", "what"]
    has_freshness_explain = any(f in query_lower for f in freshness_words) and has_and and any(e in query_lower for e in explain_words)
    
    # Decomposition indicators
    decomp_indicators = [
        has_comparison and has_and,  # "Compare X and Y"
        query_lower.count(",") >= 2,  # "pros, cons, and implications"
        ("explain" in query_lower or "discuss" in query_lower) and has_and,  # Multi-part questions
        "pros and cons" in query_lower or "benefits and drawbacks" in query_lower,  # Explicit multi-part
        has_freshness_explain,  # "Latest research and explain architecture"
    ]
    
    has_decomp_indicator = any(decomp_indicators)
    
    if not has_decomp_indicator:
        return {
            "should_decompose": False,
            "subtasks": [],
            "reasoning": "Complex query but no multi-part indicators detected"
        }
    
    # LLM-based decomposition
    decomposition_prompt = f"""Analyze this query and break it into focused sub-tasks if needed.

Query: {query}

Return valid JSON ONLY (no markdown, no extra text):{{
  "can_decompose": true,
  "subtasks": [
    {{"number": 1, "task": "first focused question", "purpose": "what this explores"}},
    {{"number": 2, "task": "second focused question", "purpose": "what this explores"}},
    {{"number": 3, "task": "third focused question", "purpose": "what this explores"}}
  ],
  "synthesis_instruction": "Intelligently combine all results: (1) DEDUPLICATE: Remove any repeated information across answers, (2) RESOLVE CONTRADICTIONS: If answers conflict, explain why and reconcile them, (3) STRUCTURE: Organize into logical sections with clear hierarchy, (4) REASON: Explain relationships and dependencies between components, (5) CONCLUDE: Provide synthesis-based recommendations or insights"
}}

Or if not decomposable:{{
  "can_decompose": false,
  "subtasks": [],
  "synthesis_instruction": ""
}}"""

    try:
        raw = call_llm(decomposition_prompt)
        import json, re
        
        # Try to extract JSON more robustly
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            json_str = json_match.group()
            # Clean up common JSON problems
            json_str = json_str.replace('\\n', ' ').replace('\\', '')
            decomp = json.loads(json_str)
            
            if decomp.get("can_decompose", False) and decomp.get("subtasks"):
                logging.info(f"🔀 Query decomposed into {len(decomp['subtasks'])} sub-tasks")
                return {
                    "should_decompose": True,
                    "subtasks": decomp["subtasks"],
                    "reasoning": f"Complex multi-part query: {len(decomp['subtasks'])} focused sub-tasks",
                    "synthesis_instruction": decomp.get("synthesis_instruction", "Combine results coherently")
                }
    except json.JSONDecodeError as e:
        logging.warning(f"⚠️ Decomposition JSON parse failed: {e}")
    except Exception as e:
        logging.warning(f"⚠️ Decomposition LLM failed: {e}")
    
    return {
        "should_decompose": False,
        "subtasks": [],
        "reasoning": "Decomposition analysis inconclusive - proceeding with single-step"
    }

# ── Step 3.2: Sub-task Executor ──────────────────────────────────────────────
def execute_subtask(subtask_dict: dict, session_id: str, parent_context: str = "") -> dict:
    """
    Executes a single sub-task using the full agent loop.
    Returns: {task: str, answer: str, success: bool, sources: list}
    """
    subtask = subtask_dict.get("task", "")
    task_num = subtask_dict.get("number", 1)
    purpose = subtask_dict.get("purpose", "")
    
    logging.info(f"\n{'─'*40}")
    logging.info(f"📍 Sub-task {task_num}: {subtask}")
    logging.info(f"   Purpose: {purpose}")
    
    # Inject parent context into subtask
    enriched_query = subtask
    if parent_context:
        enriched_query = f"{parent_context}\nSpecific question: {subtask}"
    
    # Route the subtask using smart routing
    route_info = route_sources(enriched_query)
    
    logging.info(f"   🧭 Routing: {route_info['reasoning']} (confidence: {route_info['confidence']})")
    
    # Execute tools for this subtask
    docs = execute_tools(enriched_query, route_info["primary"])
    
    if not docs:
        docs = execute_tools(enriched_query, route_info["fallback"])
    
    if not docs:
        logging.warning(f"⚠️ No documents for sub-task {task_num}")
        return {
            "task": subtask,
            "number": task_num,
            "answer": "No relevant information found",
            "success": False,
            "sources": []
        }
    
    # Rerank documents
    reranked = rerank_documents(enriched_query, docs, top_n=3)
    context = "\n\n".join([doc.page_content for doc in reranked])
    
    # Generate answer for subtask
    prompt = f"""You are a focused expert answering a specific question.

Context:
{context}

Question: {subtask}

Answer concisely and directly. Focus on answering this specific question.
Answer:"""

    answer = call_llm(prompt)
    logging.info(f"   ✅ Answer: {answer[:80]}...")
    
    sources = list(set([
        doc.metadata.get("source", "unknown") for doc in reranked
    ]))
    
    return {
        "task": subtask,
        "number": task_num,
        "purpose": purpose,
        "answer": answer,
        "success": True,
        "sources": sources
    }

# ── Step 3.3: Result Synthesizer ─────────────────────────────────────────────
def synthesize_results(original_query: str, subtask_results: list, synthesis_instruction: str) -> str:
    """
    Combines sub-task results into a coherent final answer.
    Returns: synthesized answer string
    """
    logging.info(f"\n{'═'*50}")
    logging.info(f"🔗 Synthesizing {len(subtask_results)} sub-task results...")
    
    # Build context from all subtask results
    results_context = "\n\n".join([
        f"Sub-task {r['number']}: {r['task']}\nAnswer: {r['answer']}"
        for r in subtask_results if r.get("success")
    ])
    
    synthesis_prompt = f"""You are an expert at synthesizing information from multiple sources.
Your task is to combine the following sub-task answers into a comprehensive, coherent response.

Original question: {original_query}

Sub-task results:
{results_context}

Synthesis approach: {synthesis_instruction}

Create a final answer that:
1. Directly addresses the original question
2. Integrates insights from all sub-tasks
3. Maintains logical flow and coherence
4. Highlights key relationships between sub-tasks
5. Is comprehensive yet concise

Final Answer:"""

    synthesis = call_llm(synthesis_prompt)
    logging.info(f"🔗 Synthesis complete: {synthesis[:100]}...")
    
    return synthesis

# ── MAIN AGENT LOOP ───────────────────────────────────────────────────────────
def run_agent(query: str, session_id: str = "default") -> dict:
    """
    Full agentic loop with multi-step planning:
    Decompose → Plan → Execute → Reflect → Adaptive Retry → Synthesize → Memory → Return
    
    If query is complex and multi-part, decomposes into sub-tasks.
    Each sub-task runs through independent routing/retry.
    Results are synthesized into coherent final answer.
    """
    logging.info(f"\n{'='*50}")
    logging.info(f"🤖 Agent started for: {query}")

    # Memory recall
    past_context = memory.recall(query, session_id)

    # ─── STEP 0: Multi-Step Planning (Query Decomposition) ─────────────────
    decomposition = decompose_query(query)
    
    if decomposition.get("should_decompose", False):
        logging.info(f"🔀 Using multi-step planning strategy")
        
        subtasks = decomposition.get("subtasks", [])
        synthesis_instruction = decomposition.get("synthesis_instruction", "")
        
        # Execute each sub-task
        subtask_results = []
        for subtask_dict in subtasks:
            result = execute_subtask(subtask_dict, session_id, parent_context=query)
            subtask_results.append(result)
        
        # Synthesize results
        final_answer = synthesize_results(query, subtask_results, synthesis_instruction)
        
        # Collect all sources from subtasks
        all_sources = []
        for result in subtask_results:
            all_sources.extend(result.get("sources", []))
        all_sources = list(set(all_sources))
        
        # Memory save for main query
        memory.save(query, final_answer, session_id)
        
        logging.info(f"✅ Multi-step agent complete | {len(subtasks)} sub-tasks executed")
        
        return {
            "answer": final_answer,
            "sources": all_sources,
            "attempts": 1,
            "plan": {
                "source": "multi-step",
                "needs_rewrite": False,
                "reasoning_depth": "deep",
                "route_info": {"confidence": 0.9}
            },
            "reflection": {
                "good": True,
                "reason": "multi-step synthesis",
                "complexity": "complex"
            },
            "search_type": "multi-step",
            "subtask_results": subtask_results,
            "is_multi_step": True
        }
    
    # ─── SINGLE-STEP PATH (Original agent loop) ──────────────────────────
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
        "search_type": source,
        "is_multi_step": False
    }