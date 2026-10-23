"""
core/synthesizer.py — Result Synthesis Engine
-----------------------------------------------
Combines results from multiple sub-tasks or tool calls into
a coherent final answer.

Used by:
  - Multi-step planning (subtask results)
  - Tool chaining (multi-tool results)

Usage:
    from core.synthesizer import synthesize_results
    final = synthesize_results(query, subtask_results, instruction)
"""

import logging
from app.config import call_llm

logger = logging.getLogger(__name__)


def synthesize_results(
    original_query: str,
    results: list[dict],
    synthesis_instruction: str = "",
) -> str:
    """
    Combine multiple sub-task/tool results into a coherent answer.

    Args:
        original_query: The original user question
        results: List of dicts with {task/tool, answer/output, number, purpose, success, sources}
        synthesis_instruction: How to combine results

    Returns:
        Synthesized answer string
    """
    logger.info(f"🔗 Synthesizing {len(results)} results...")

    successful = [r for r in results if r.get("success", True)]

    if not successful:
        return "Unable to find sufficient information to answer this question."

    # Build context from all results
    results_context = "\n\n".join([
        f"Part {r.get('number', i+1)}: {r.get('task', r.get('tool', 'Unknown'))}\n"
        f"Answer: {r.get('answer', r.get('output', 'No result'))}"
        for i, r in enumerate(successful)
    ])

    if not synthesis_instruction:
        synthesis_instruction = (
            "Deduplicate, resolve contradictions, structure logically, "
            "and provide a clear, comprehensive answer."
        )

    prompt = f"""You are an expert at synthesizing information from multiple sources.
Your task is to combine the following results into a comprehensive, coherent response.

Original question: {original_query}

Results:
{results_context}

Synthesis approach: {synthesis_instruction}

Create a final answer that:
1. Directly addresses the original question
2. Integrates insights from all parts
3. Maintains logical flow and coherence
4. Removes redundancy
5. Is comprehensive yet concise

Final Answer:"""

    synthesis = call_llm(prompt)
    logger.info(f"🔗 Synthesis complete: {synthesis[:100]}...")

    return synthesis


def synthesize_tool_chain(
    original_query: str,
    step_results: list[dict],
) -> str:
    """
    Synthesize results from a tool chain execution.
    Similar to synthesize_results but formatted for tool step outputs.

    Args:
        original_query: The original user question
        step_results: List of {step, tool, purpose, output, status}

    Returns:
        Synthesized answer string
    """
    completed = [s for s in step_results if s.get("status") == "done"]

    if not completed:
        return "No results to combine — all steps failed."

    results_context = "\n\n".join([
        f"Step {s['step']} [{s['tool']}] — {s.get('purpose', '')}:\n{s.get('output', '')}"
        for s in completed
    ])

    prompt = f"""You are an expert at synthesizing information from multiple sources and tools.

Original question: {original_query}

Tool execution results:
{results_context}

Create a comprehensive final answer that:
1. Directly answers the original question
2. Integrates all tool results coherently
3. Removes redundancy
4. Is well-structured and clear
5. Cites which tool provided which insight where relevant

Final Answer:"""

    synthesis = call_llm(prompt)
    logger.info(f"🔗 Tool chain synthesis complete: {synthesis[:100]}...")

    return synthesis
