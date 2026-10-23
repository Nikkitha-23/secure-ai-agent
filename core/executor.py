"""
core/executor.py — Tool & Retrieval Execution Engine
------------------------------------------------------
Handles the actual execution of tools, retrieval, and LLM answer generation.
This is the "hands" of the agent — it does what the planner decided.

Responsibilities:
  - Execute tool calls via the tool registry
  - Run retrieval (vector + web search)
  - Rerank retrieved documents
  - Generate LLM answers from context
  - Execute sub-tasks for multi-step plans

Usage:
    from core.executor import Executor
    executor = Executor()
    docs = executor.retrieve(query, source="pdf")
    answer = executor.generate_answer(query, context, memory_context)
"""

import logging
from typing import Optional
from app.config import settings, call_llm
from tools.registry import tool_registry, ToolCallResult

logger = logging.getLogger(__name__)


class Executor:
    """
    Execution engine for the agent.
    Handles retrieval, tool execution, and answer generation.
    """

    def __init__(self):
        self._retriever = None
        self._retriever_initialized = False

    def _get_retriever(self):
        """Lazy-load the unified hybrid retriever."""
        if not self._retriever_initialized:
            try:
                from rag.hybrid_retrieval import get_hybrid_retriever
                self._retriever = get_hybrid_retriever()
                self._retriever_initialized = True
            except Exception as e:
                logger.error(f"❌ Hybrid retriever initialization failed: {e}")
                try:
                    from rag.retrieve import get_retriever
                    self._retriever = get_retriever()
                except Exception as ex:
                    logger.error(f"❌ Fallback retriever failed: {ex}")
                    self._retriever = None
                self._retriever_initialized = True
        return self._retriever

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, source: str = "pdf") -> list:
        """
        Retrieve documents based on source strategy (Hybrid RAG for PDF, Web search for web).

        Args:
            query: Search query
            source: "pdf" | "web" | "both"

        Returns:
            List of Document objects
        """
        all_docs = []

        if source in ("pdf", "both"):
            retriever = self._get_retriever()
            if retriever:
                try:
                    pdf_docs = retriever.invoke(query)
                    logger.info(f"📄 Hybrid PDF chunks retrieved: {len(pdf_docs)}")
                    all_docs.extend(pdf_docs)
                except Exception as e:
                    logger.error(f"❌ Hybrid PDF retrieval failed: {e}")

        if source in ("web", "both"):
            try:
                from rag.web_search import web_search
                web_docs = web_search(query, max_results=3)
                logger.info(f"🌐 Web results retrieved: {len(web_docs)}")
                all_docs.extend(web_docs)
            except Exception as e:
                logger.error(f"❌ Web search failed: {e}")

        return all_docs

    def rerank(self, query: str, docs: list, top_n: int = None) -> list:
        """Rerank documents using cross-encoder."""
        top_n = top_n or settings.RERANKER_TOP_N
        if not docs:
            return []
        try:
            from rag.reranker import rerank_documents
            reranked = rerank_documents(query, docs, top_n=top_n)
            logger.info(f"🏆 Reranked: top {len(reranked)} of {len(docs)} chunks")
            return reranked
        except Exception as e:
            logger.error(f"❌ Reranking failed: {e}")
            return docs[:top_n]

    def validate_evidence(self, query: str, docs: list):
        """Validate retrieved document evidence quality and sufficiency."""
        try:
            from rag.evidence import validate_evidence
            return validate_evidence(query, docs)
        except Exception as e:
            logger.error(f"⚠️ Evidence validation failed: {e}")
            return None

    # ── Tool Execution ───────────────────────────────────────────────────────

    def execute_tool(
        self,
        tool_name: str,
        args: dict = None,
        user_role: str = "public",
    ) -> ToolCallResult:
        """Execute a tool via the tool registry."""
        return tool_registry.execute(tool_name, args, user_role)

    # ── Answer Generation ────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        context: str,
        memory_context: str = "",
        source_type: str = "pdf",
    ) -> str:
        """
        Generate an answer from retrieved context using LLM.

        Args:
            query: User question
            context: Retrieved/reranked document context
            memory_context: Memory context to inject
            source_type: "pdf" (strict) or "web" (synthesis mode)

        Returns:
            Generated answer string
        """
        memory_section = f"\nUser Preferences & Past Context:\n{memory_context}\n" if memory_context else ""

        if source_type == "pdf":
            prompt = f"""You are a document question-answering system.
{memory_section}
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
{query}

Answer (use only context facts):"""
        else:
            prompt = f"""You are a helpful AI assistant answering using retrieved information.
{memory_section}
GUIDELINES:
- Synthesize a clear, direct answer using the information in the context below.
- You may paraphrase and combine facts from multiple sources.
- Stick to what the context supports — don't invent specifics not present in it.
- Only say "Not found in documents." if the context truly has nothing relevant.
{memory_section}
Context:
{context}

Question:
{query}

Answer:"""

        answer = call_llm(
            prompt,
            system_prompt=f"You are a secure and helpful AI assistant. {memory_section}",
        )
        return answer

    # ── Sub-task Execution ───────────────────────────────────────────────────

    def execute_subtask(
        self,
        subtask: dict,
        session_id: str = "default",
        parent_context: str = "",
    ) -> dict:
        """
        Execute a single sub-task from a decomposed query.

        Args:
            subtask: {number, task, purpose}
            session_id: Session ID for memory
            parent_context: Original query for context enrichment

        Returns:
            {task, number, purpose, answer, success, sources}
        """
        task = subtask.get("task", "")
        task_num = subtask.get("number", 1)
        purpose = subtask.get("purpose", "")

        logger.info(f"\n{'─'*40}")
        logger.info(f"📍 Sub-task {task_num}: {task}")

        enriched_query = f"{parent_context}\nSpecific question: {task}" if parent_context else task

        # Check if a specific tool should handle this
        tool_name = subtask.get("tool")
        if not tool_name or tool_name in ("general", "both", "pdf", "web"):
            from core.planner import detect_intent, IntentType
            intent = detect_intent(task)
            tool_name = tool_registry.select_tool(intent, task)

        if tool_name and tool_name not in ("general", "both", "pdf", "web"):
            # Build appropriate kwargs based on tool schema
            sub_query = subtask.get("query") or task
            args = {"query": sub_query, "expression": sub_query, "code": sub_query, "filename": sub_query}
            result = self.execute_tool(tool_name, args, user_role="public")
            if result.success:
                return {
                    "task": task, "number": task_num, "purpose": purpose,
                    "answer": result.output, "success": True,
                    "sources": [f"{tool_name.replace('_', ' ').title()} Tool"],
                }
            else:
                logger.warning(f"⚠️ Tool {tool_name} failed for subtask {task_num}: {result.error}")
                return {
                    "task": task, "number": task_num, "purpose": purpose,
                    "answer": f"Tool '{tool_name}' failed: {result.error}",
                    "success": False,
                    "sources": [],
                }

        # Default: Unified Hybrid RAG path
        from core.planner import route_sources
        route_info = route_sources(enriched_query)

        docs = self.retrieve(enriched_query, route_info["primary"])
        if not docs and route_info["primary"] != route_info["fallback"]:
            docs = self.retrieve(enriched_query, route_info["fallback"])

        if not docs:
            logger.warning(f"⚠️ No documents for sub-task {task_num}")
            return {
                "task": task, "number": task_num, "purpose": purpose,
                "answer": "No relevant information found in documents.",
                "success": False, "sources": [],
            }

        reranked = self.rerank(enriched_query, docs)
        reranked_docs = [d[0] if isinstance(d, tuple) else d for d in reranked]
        context = "\n\n".join([doc.page_content for doc in reranked_docs])

        answer = call_llm(
            f"""You are a focused expert answering a specific question.

Context:
{context}

Question: {task}

Answer concisely and directly using only facts from the context.
Answer:"""
        )

        from rag.evidence import format_source_attribution
        sources = format_source_attribution(reranked_docs)

        logger.info(f"   ✅ Sub-task {task_num} complete: {answer[:80]}...")

        return {
            "task": task, "number": task_num, "purpose": purpose,
            "answer": answer, "success": True, "sources": list(set([str(s) for s in sources])),
        }


# ── Top-level Helper Functions ───────────────────────────────────────────────

def generate_direct_answer(query: str, memory_context: str = "") -> str:
    prompt = f"User Question: {query}"
    if memory_context:
        prompt = f"User Past Memory:\n{memory_context}\n\n{prompt}"
    return call_llm(prompt)


def execute_plan_step(subtask: dict, user_role: str = "student", user_id: str = "default_user") -> dict:
    executor = Executor()
    tool_name = subtask.get("tool", "direct_llm")
    query = subtask.get("query") or subtask.get("description") or ""

    if tool_name and tool_name not in ("direct_llm", "pdf", "web", "both", "general"):
        # Map arguments safely according to tool requirement
        args = {"query": query, "expression": query, "code": query, "filename": query}
        res = executor.execute_tool(tool_name, args, user_role=user_role)
        if res.success:
            return {"result": str(res.output), "success": True, "sources": [tool_name]}
        else:
            return {"result": f"Tool execution failed: {res.error}", "success": False, "sources": []}

    # Retrieval path (PDF / Web / Both)
    source_type = "pdf" if tool_name == "pdf" else ("web" if tool_name == "web" else "both")
    docs = executor.retrieve(query, source=source_type)
    if docs:
        reranked = executor.rerank(query, docs)
        reranked_docs = [d[0] if isinstance(d, tuple) else d for d in reranked]
        context = "\n\n".join([d.page_content for d in reranked_docs])
        answer = executor.generate_answer(query, context, source_type=source_type)
        from rag.evidence import format_source_attribution
        sources = format_source_attribution(reranked_docs)
        return {"result": answer, "success": True, "sources": list(dict.fromkeys([str(s) for s in sources]))}

    if tool_name in ("pdf", "web", "both"):
        return {"result": "Not found in documents.", "success": False, "sources": []}

    return {"result": generate_direct_answer(query), "success": True, "sources": []}


def execute_multi_step_plan(plan, workspace, max_steps: int = 10, user_role: str = "student", user_id: str = "default_user") -> dict:
    executor = Executor()
    subtasks = plan.subtasks if hasattr(plan, "subtasks") else plan.get("subtasks", [])

    if hasattr(plan, "to_dict"):
        workspace.plan = plan.to_dict()
    elif isinstance(plan, dict):
        workspace.plan = plan
    workspace.start()

    results = []
    all_sources = []
    steps_taken = 0

    for i, st in enumerate(subtasks):
        if steps_taken >= max_steps or len(workspace.steps) >= max_steps:
            logger.warning(f"⚠️ Reached maximum step limit ({max_steps}). Stopping execution.")
            break

        step_num = i + 1
        tool_name = st.get("tool", "general")
        desc = st.get("description") or st.get("query") or f"Subtask {step_num}"
        query_val = st.get("query") or desc

        # Prevent executing the same step twice
        existing = [s for s in workspace.steps if s.step_num == step_num and s.status in ("completed", "done")]
        if existing:
            logger.info(f"Skipping already completed step {step_num}")
            continue

        wstep = workspace.add_step(tool_name, desc, step_num=step_num)
        wstep.start()

        parent_ctx = "\n".join([f"Step {r.get('number', i+1)}: {r.get('answer', '')}" for r in results if r.get("answer")])
        subtask_dict = {"number": step_num, "task": desc, "purpose": desc, "tool": tool_name, "query": query_val}
        res = executor.execute_subtask(subtask_dict, session_id=user_id, parent_context=parent_ctx)

        if res.get("success"):
            wstep.complete(res.get("answer", ""))
            results.append(res)
            all_sources.extend(res.get("sources", []))
        else:
            wstep.fail(res.get("answer", "Failed step"))
            results.append(res)  # Preserve failure in results so subsequent steps/synthesis have context

        steps_taken += 1

    from core.synthesizer import synthesize_results
    query_str = plan.original_query if hasattr(plan, "original_query") else ""
    final_ans = synthesize_results(query_str, results)

    if workspace.status in ["running", "pending", "planning"]:
        workspace.complete(final_ans, search_type="multi_step", confidence=0.85)

    return {
        "answer": final_ans,
        "sources": list(set(all_sources)),
        "steps_taken": steps_taken,
        "confidence": 0.85
    }
