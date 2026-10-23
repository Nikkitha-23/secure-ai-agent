"""
core/agent.py — Canonical Core Agent Loop
------------------------------------------
The primary agentic control loop for Secure AI Agent.

Execution Lifecycle:
  1. Security Validation (Prompt injection / PII filter)
  2. Short-term & Long-term Memory Recall
  3. Intent & Complexity Detection (via Planner)
  4. Task Decomposition / Planning (if complex/multi-step)
  5. Dynamic Tool Selection & Execution (via Tool Registry & Executor)
  6. Multi-step Execution & Observation Loop (with Max Step Guard)
  7. Reflection & Quality Assessment (Lightweight / LLM-based)
  8. Re-planning if evidence is insufficient/hallucinated
  9. Answer Synthesis
 10. Long-term Memory Update
 11. Final Structured Response Formatting
"""

import time
import logging
import traceback
from typing import Dict, Any, List, Optional

from app.config import settings, call_llm
from app.models import AgentResponse, SourceInfo, StepLog
from core.planner import plan_query, AgentPlan
from core.executor import execute_plan_step, generate_direct_answer, execute_multi_step_plan
from core.reflector import reflect_on_answer, ReflectionResult
from core.synthesizer import synthesize_results
from core.workspace import get_or_create_workspace, AgentWorkspace

# Import RAG & Security components safely with fallbacks
try:
    from security.filter import check_input
    SECURITY_AVAILABLE = True
    def filter_input(text: str, role: str = "student"):
        res = check_input(text)
        if res.get("status") == "block":
            return {"safe": False, "risk_score": 1.0, "reason": res.get("reason", "Prompt injection or policy violation detected.")}
        return {"safe": True, "risk_score": 0.0, "reason": "ok"}
except ImportError:
    SECURITY_AVAILABLE = False
    def filter_input(text: str, role: str = "student"):
        return {"safe": True, "risk_score": 0.0, "reason": "No security module"}


try:
    from rag.memory import get_memory_manager
    MEMORY_AVAILABLE = True
    memory_manager = get_memory_manager()
except Exception as e:
    MEMORY_AVAILABLE = False
    memory_manager = None

try:
    from hallucination_guard import check_hallucination
    HALLUCINATION_GUARD_AVAILABLE = True
except ImportError:
    HALLUCINATION_GUARD_AVAILABLE = False
    def check_hallucination(answer: str, context: str):
        return {"has_hallucination": False, "score": 1.0}

logger = logging.getLogger(__name__)


class SecureAgent:
    """
    Unified canonical agent engine coordinating planning, tool execution,
    reflection, memory, and safety.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id

    def run(
        self,
        query: str,
        user_role: str = "student",
        user_id: str = "default_user",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main execution entry point for agent queries.
        Returns a dict adhering to AgentResponse schema.
        """
        start_time = time.time()
        active_session = session_id or self.session_id or f"sess_{user_id}_{int(start_time)}"
        workspace = get_or_create_workspace(active_session, query)

        step_logs: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []

        # Reset tool call counts for new request
        from tools.registry import tool_registry
        tool_registry.reset_call_counts()

        # ── Step 1: Security Validation ───────────────────────────────────────
        workspace.note("Step 1: Running security input validation...")
        if SECURITY_AVAILABLE:
            sec_check = filter_input(query, role=user_role)
            if not sec_check.get("safe", True):
                reason = sec_check.get("reason", "Prompt injection or policy violation detected.")
                workspace.fail(reason)
                return {
                    "answer": f"⚠️ Security Alert: Query blocked. Reason: {reason}",
                    "confidence": 0.0,
                    "search_type": "security_blocked",
                    "sources": [],
                    "steps": [
                        {
                            "step_num": 1,
                            "tool": "security_filter",
                            "purpose": "Input security validation",
                            "status": "failed",
                            "result": reason,
                            "duration_ms": int((time.time() - start_time) * 1000)
                        }
                    ],
                    "total_duration_ms": int((time.time() - start_time) * 1000),
                    "task_id": workspace.task_id,
                    "session_id": active_session
                }

        # ── Step 2: Memory Recall ─────────────────────────────────────────────
        memory_context = ""
        if MEMORY_AVAILABLE and memory_manager:
            try:
                recalled = memory_manager.recall(query=query, session_id=user_id)
                if isinstance(recalled, list) and recalled:
                    memory_context = "\n".join([str(m) for m in recalled])
                elif isinstance(recalled, str):
                    memory_context = recalled
                if memory_context:
                    workspace.note(f"Memory recalled for user {user_id}")
            except Exception as e:
                logger.warning(f"Memory recall error: {e}")

        # ── Step 3: Intent & Complexity Planning ─────────────────────────────
        workspace.note("Step 2: Analyzing intent and constructing execution plan...")
        plan: AgentPlan = plan_query(query, memory_context=memory_context)
        workspace.note(f"Planned route: {plan.search_type}, complexity: {plan.complexity_level}")

        final_answer = ""
        search_type = plan.search_type
        confidence = 0.85
        total_steps = 0

        # ── Step 4: Execution Strategy ────────────────────────────────────────
        try:
            if plan.search_type == "direct_llm" or plan.intent == "greeting":
                workspace.note("Executing direct LLM answer path (no retrieval required)...")
                step = workspace.add_step("direct_llm", "Generate direct answer")
                step.start()

                final_answer = generate_direct_answer(query, memory_context)
                step.complete(final_answer[:200])
                sources = []
                total_steps = 1

            elif plan.subtasks and len(plan.subtasks) > 1:
                workspace.note(f"Executing multi-step plan with {len(plan.subtasks)} subtasks...")
                exec_result = execute_multi_step_plan(
                    plan=plan,
                    workspace=workspace,
                    max_steps=settings.MAX_AGENT_STEPS,
                    user_role=user_role,
                    user_id=user_id
                )
                final_answer = exec_result.get("answer", "")
                sources.extend(exec_result.get("sources", []))
                total_steps = exec_result.get("steps_taken", len(plan.subtasks))
                confidence = exec_result.get("confidence", 0.85)

            else:
                workspace.note(f"Executing single tool/retrieval task via tool: {plan.recommended_tool}...")
                step = workspace.add_step(plan.recommended_tool, f"Execute query via {plan.search_type}")
                step.start()

                subtask = plan.subtasks[0] if plan.subtasks else {
                    "step_num": 1,
                    "tool": plan.recommended_tool,
                    "description": query,
                    "query": query
                }

                res = execute_plan_step(subtask, user_role=user_role, user_id=user_id)
                if res.get("success", True):
                    step.complete(res.get("result", ""))
                else:
                    step.fail(res.get("result", "Execution failed"))

                final_answer = res.get("result", "")
                if res.get("sources"):
                    sources.extend(res.get("sources", []))
                total_steps = 1

            # ── Step 5: Reflection & Validation ───────────────────────────────
            workspace.note("Step 3: Reflecting on response quality...")
            reflection: ReflectionResult = reflect_on_answer(
                query=query,
                answer=final_answer,
                sources=sources,
                complexity=plan.complexity_level
            )

            if not reflection.is_valid and reflection.suggested_action == "replan":
                workspace.note(f"Reflection critique: {reflection.feedback}. Attempting re-plan synthesis...")
                # Single-pass recovery fallback
                recovery_prompt = (
                    f"Original Question: {query}\n"
                    f"Initial Draft Answer: {final_answer}\n"
                    f"Critique / Missing Info: {reflection.feedback}\n\n"
                    f"Please produce a complete, accurate, and self-corrected final response."
                )
                final_answer = call_llm(recovery_prompt)
                confidence = max(0.6, confidence - 0.15)

            # ── Step 6: Memory Storage ────────────────────────────────────────
            if MEMORY_AVAILABLE and memory_manager and final_answer:
                try:
                    memory_manager.save(question=query, answer=final_answer, session_id=user_id)
                    workspace.note(f"Stored conversation in memory for user {user_id}")
                except Exception as e:
                    logger.warning(f"Memory store error: {e}")

            workspace.complete(final_answer, search_type=search_type, confidence=confidence)

        except Exception as e:
            logger.error(f"Agent execution error: {e}\n{traceback.format_exc()}")
            workspace.fail(str(e))
            final_answer = f"I encountered an error while processing your request: {str(e)}"
            confidence = 0.0

        # Construct StepLogs from workspace steps (respecting max steps)
        max_steps_allowed = getattr(settings, "MAX_AGENT_STEPS", 10)
        formatted_steps = []
        for s in workspace.steps[:max_steps_allowed]:
            formatted_steps.append({
                "step_num": s.step_num,
                "tool": s.tool,
                "purpose": s.purpose,
                "status": s.status,
                "result": s.result[:300] if s.result else "",
                "duration_ms": s.duration_ms
            })

        duration_ms = int((time.time() - start_time) * 1000)

        return {
            "answer": final_answer,
            "confidence": confidence,
            "search_type": search_type,
            "sources": sources,
            "steps": formatted_steps,
            "total_duration_ms": duration_ms,
            "task_id": workspace.task_id,
            "session_id": active_session
        }

# Global singleton helper
_agent_instance = SecureAgent()

def run_agent(
    query: str,
    user_role: str = "student",
    user_id: str = "default_user",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to run the canonical agent loop."""
    return _agent_instance.run(query=query, user_role=user_role, user_id=user_id, session_id=session_id)
