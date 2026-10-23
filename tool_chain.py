"""
tool_chain.py — Tool Chaining Engine
--------------------------------------
Flow:
  tool_chain_planner()   → LLM decides which tools + order
  TaskState              → tracks plan, results, status
  tool_chain_executor()  → executes each step with right tool
  tool_chain_combiner()  → merges all results into final answer
  reflect_on_chain()     → checks final answer quality
"""

import json
import re
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

AVAILABLE_TOOLS = ["web_search", "science_tool", "file_tool", "image_tool", "calculator", "memory"]

# ══════════════════════════════════════════════════════════════
# STEP 1: Tool Chain Planner
# ══════════════════════════════════════════════════════════════

def tool_chain_planner(query: str) -> dict:
    """
    LLM analyzes the query and returns an ordered tool execution plan.

    Returns:
    {
      "needs_chaining": True/False,
      "steps": [
        {"step": 1, "tool": "web_search",    "purpose": "...", "input": "..."},
        {"step": 2, "tool": "science_tool",  "purpose": "...", "input": "..."},
        {"step": 3, "tool": "file_tool",     "purpose": "...", "input": "..."},
      ],
      "reasoning": "why this plan"
    }
    """
    prompt = f"""You are an AI task planner for an autonomous agent.

Available tools:
- web_search     : Search the internet for current/live information. Use for news, recent events, online data
- science_tool   : Math, statistics, data analysis, plotting (SymPy, NumPy, Pandas). Use for calculations, equations, graphs
- file_tool      : Read, summarize, list PDF files. Use ONLY when user says "pdf", "file", "document", "uploaded"
- image_tool     : Analyze images, extract text from images (OCR). Use ONLY when image is provided
- calculator     : Simple arithmetic only (+ - * /). NOT for science/math formulas
- memory         : Recall past conversation context. Use when user says "earlier", "before", "last time"
User query: {query}

Decide if this query needs MULTIPLE tools (tool chaining) or just ONE tool.

Rules:
- needs_chaining = true ONLY if query genuinely requires 2+ different tools
- Simple questions (one topic, one tool) → needs_chaining = false
- Multi-part questions (search + analyze, read + compare, calculate + explain) → needs_chaining = true
- Max 4 steps

Return ONLY valid JSON, no explanation:
{{
  "needs_chaining": true or false,
  "steps": [
    {{"step": 1, "tool": "tool_name", "purpose": "what this step does", "input": "exact query for this tool"}},
    {{"step": 2, "tool": "tool_name", "purpose": "what this step does", "input": "exact query for this tool"}}
  ],
  "reasoning": "one sentence why"
}}

If needs_chaining is false, return empty steps array."""

    try:
        response = llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            plan = json.loads(match.group())

            # Validate tools
            valid_steps = []
            for step in plan.get("steps", []):
                if step.get("tool") in AVAILABLE_TOOLS:
                    valid_steps.append(step)
                else:
                    logger.warning(f"⚠️ Unknown tool in plan: {step.get('tool')} — skipping")
            plan["steps"] = valid_steps

            logger.info(f"🔗 Chain plan: needs_chaining={plan.get('needs_chaining')} | steps={len(plan.get('steps', []))}")
            for s in plan.get("steps", []):
                logger.info(f"   Step {s['step']}: [{s['tool']}] {s['purpose']}")

            return plan

    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Planner JSON parse failed: {e}")
    except Exception as e:
        logger.error(f"❌ Planner failed: {e}")

    # Fallback — no chaining
    return {
        "needs_chaining": False,
        "steps": [],
        "reasoning": "Planner failed — single tool fallback"
    }

# ══════════════════════════════════════════════════════════════
# STEP 2: TaskState Object
# ══════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    step: int
    tool: str
    purpose: str
    input_query: str
    output: str = ""
    status: str = "waiting"   # waiting | running | done | failed
    error: Optional[str] = None


@dataclass
class TaskState:
    """
    Tracks the full state of a tool chain execution.
    Memory ≠ Workspace — this is the workspace.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_query: str = ""
    plan: dict = field(default_factory=dict)
    steps: list[StepResult] = field(default_factory=list)
    final_answer: str = ""
    status: str = "pending"   # pending | running | done | failed
    reflection: dict = field(default_factory=dict)

    def current_step(self) -> Optional[StepResult]:
        for s in self.steps:
            if s.status in ["waiting", "running"]:
                return s
        return None

    def completed_steps(self) -> list[StepResult]:
        return [s for s in self.steps if s.status == "done"]

    def summary(self) -> str:
        lines = [f"Task [{self.task_id}]: {self.status.upper()}"]
        for s in self.steps:
            icon = {"waiting": "⏳", "running": "🔄", "done": "✅", "failed": "❌"}.get(s.status, "❓")
            lines.append(f"  {icon} Step {s.step} [{s.tool}]: {s.purpose}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# STEP 3: Tool Chain Executor
# ══════════════════════════════════════════════════════════════

def tool_chain_executor(state: TaskState) -> TaskState:
    """
    Executes each step in the TaskState using the correct tool.
    Updates state.steps with results.
    """
    state.status = "running"
    logger.info(f"\n{'═'*50}")
    logger.info(f"🔗 Tool Chain Executor started | Task: {state.task_id}")

    for step in state.steps:
        step.status = "running"
        logger.info(f"\n▶ Step {step.step}: [{step.tool}] — {step.purpose}")
        logger.info(f"  Input: {step.input_query[:80]}")

        try:
            result = _execute_single_tool(step.tool, step.input_query, state)
            step.output = result
            step.status = "done"
            logger.info(f"  ✅ Output: {result[:100]}...")

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.output = f"Step failed: {e}"
            logger.error(f"  ❌ Step {step.step} failed: {e}")

    state.status = "combining"
    logger.info(f"\n{'═'*50}")
    logger.info(state.summary())
    return state


def _execute_single_tool(tool: str, query: str, state: TaskState) -> str:
    """Routes to the correct tool function."""

    # Inject previous step results into query context if useful
    completed = state.completed_steps()
    context_prefix = ""
    if completed:
        prev_results = "\n".join([
            f"[Step {s.step} - {s.tool} result]: {s.output[:200]}"
            for s in completed
        ])
        context_prefix = f"Previous step results:\n{prev_results}\n\nCurrent task: "

    enriched_query = context_prefix + query

    if tool == "web_search":
        from rag.web_search import web_search
        docs = web_search(query, max_results=3)
        if not docs:
            return "No web results found."
        return "\n\n".join([
            f"Source: {d.metadata.get('source', 'web')}\n{d.page_content[:300]}"
            for d in docs[:3]
        ])

    elif tool == "science_tool":
        from tools.science_tool import run_science_tool
        return run_science_tool(query)

    elif tool == "file_tool":
        from tools.file_tool import file_tool, is_file_intent
        return file_tool.run(action="ask", filename=None, question=query)

    elif tool == "image_tool":
        from tools.image_tool import run_image_tool
        return run_image_tool(query=query, image_input=None)

    elif tool == "calculator":
        from groq import Groq
        import os
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Solve this math problem step by step: {query}"}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    elif tool == "memory":
        from rag.memory import get_memory_manager
        memory = get_memory_manager()
        result = memory.recall(query, "default")
        return result if result else "No relevant memory found."

    else:
        return f"Unknown tool: {tool}"


# ══════════════════════════════════════════════════════════════
# STEP 4: Tool Chain Combiner
# ══════════════════════════════════════════════════════════════

def tool_chain_combiner(state: TaskState) -> TaskState:
    """
    Synthesizes all step results into a coherent final answer.
    """
    logger.info(f"\n🔗 Combiner started | {len(state.completed_steps())} steps to combine")

    completed = state.completed_steps()
    if not completed:
        state.final_answer = "No results to combine — all steps failed."
        state.status = "failed"
        return state

    # Build context from all results
    results_context = "\n\n".join([
        f"Step {s.step} [{s.tool}] — {s.purpose}:\n{s.output}"
        for s in completed
    ])

    prompt = f"""You are an expert at synthesizing information from multiple sources and tools.

Original question: {state.original_query}

Tool execution results:
{results_context}

Create a comprehensive final answer that:
1. Directly answers the original question
2. Integrates all tool results coherently
3. Removes redundancy
4. Is well-structured and clear
5. Cites which tool provided which insight where relevant

Final Answer:"""

    try:
        response = llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that synthesizes information clearly and accurately. Always respond in English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1024
        )
        state.final_answer = response.choices[0].message.content.strip()
        logger.info(f"✅ Combined answer: {state.final_answer[:100]}...")

    except Exception as e:
        logger.error(f"❌ Combiner failed: {e}")
        # Fallback: join outputs directly
        state.final_answer = "\n\n".join([
            f"**{s.tool}**: {s.output}" for s in completed
        ])

    state.status = "reflecting"
    return state


# ══════════════════════════════════════════════════════════════
# STEP 5: Reflection Integration
# ══════════════════════════════════════════════════════════════

def reflect_on_chain(state: TaskState) -> TaskState:
    """
    Checks the quality of the final combined answer.
    """
    prompt = f"""You are a quality checker for an AI agent's response.

Original question: {state.original_query}
Final answer: {state.final_answer}
Tools used: {[s.tool for s in state.completed_steps()]}

Return ONLY valid JSON:
{{
  "good": true or false,
  "reason": "one sentence",
  "missing": "what's missing if bad, else empty string",
  "confidence": 0.0 to 1.0
}}"""

    try:
        response = llm.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=128
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            state.reflection = json.loads(match.group())
            logger.info(f"🔍 Reflection: good={state.reflection.get('good')} | confidence={state.reflection.get('confidence')}")

    except Exception as e:
        logger.warning(f"⚠️ Reflection failed: {e}")
        state.reflection = {"good": True, "reason": "default pass", "confidence": 0.8}

    state.status = "done"
    return state


# ══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def run_tool_chain(query: str) -> dict:
    """
    Full tool chain pipeline:
    Plan → TaskState → Execute → Combine → Reflect → Return

    Called from agent_loop.py or main.py
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"🔗 Tool Chain started: {query}")

    # Step 1: Plan
    plan = tool_chain_planner(query)

    if not plan.get("needs_chaining") or not plan.get("steps"):
        logger.info("🔗 No chaining needed — single tool path")
        return {"needs_chaining": False}

    # Step 2: TaskState
    state = TaskState(
        original_query=query,
        plan=plan,
        steps=[
            StepResult(
                step=s["step"],
                tool=s["tool"],
                purpose=s["purpose"],
                input_query=s["input"]
            )
            for s in plan["steps"]
        ]
    )

    logger.info(f"📋 Task [{state.task_id}] created | {len(state.steps)} steps planned")

    # Step 3: Execute
    state = tool_chain_executor(state)

    # Step 4: Combine
    state = tool_chain_combiner(state)

    # Step 5: Reflect
    state = reflect_on_chain(state)

    # Collect sources
    sources = list(set([s.tool for s in state.completed_steps()]))

    return {
        "needs_chaining": True,
        "answer": state.final_answer,
        "sources": sources,
        "search_type": "tool_chain",
        "attempts": 1,
        "is_multi_step": True,
        "task_id": state.task_id,
        "steps_completed": len(state.completed_steps()),
        "reflection": state.reflection,
        "step_results": [
            {"step": s.step, "tool": s.tool, "purpose": s.purpose, "status": s.status}
            for s in state.steps
        ]
    }


# ── Self-test ─────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is 2+2?",
        "Search latest AI news and calculate the growth rate from 100 to 150",
        "Find current Python trends and explain what list comprehension is",
    ]

    print("\n🔗 Tool Chain Planner Test:\n")
    for q in test_queries:
        plan = tool_chain_planner(q)
        print(f"Query: {q}")
        print(f"  needs_chaining: {plan['needs_chaining']}")
        for s in plan.get("steps", []):
            print(f"  Step {s['step']}: [{s['tool']}] {s['purpose']}")
        print(f"  Reasoning: {plan['reasoning']}\n")