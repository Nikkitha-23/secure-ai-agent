"""
core/reflector.py — Reflection & Re-planning Engine
------------------------------------------------------
Merged from rag/reflection.py and agent_loop.py:reflect().

Two levels of reflection:
  1. Lightweight critic (no LLM) — fast, catches obvious failures
  2. LLM-based reflection — deeper quality assessment with complexity awareness

Usage:
    from core.reflector import Reflector
    reflector = Reflector()
    verdict = reflector.reflect(query, answer, sources, complexity)
"""

import json
import re
import logging
from typing import Callable, Optional

from app.config import call_llm

logger = logging.getLogger(__name__)


# ── Lightweight Critic ───────────────────────────────────────────────────────

NOT_FOUND_MARKERS = [
    "not found in documents",
    "no relevant information found",
    "i don't know",
    "cannot find",
    "i'm unable to find",
    "no information available",
]


def quick_critique(query: str, answer: str, has_sources: bool) -> dict:
    """
    Fast heuristic critic — no LLM call.
    Catches obvious failure modes before escalating to LLM-as-judge.

    Returns:
        {"accept": bool, "reason": str}
    """
    answer_lower = (answer or "").strip().lower()

    if not answer_lower:
        return {"accept": False, "reason": "empty_answer"}

    if any(marker in answer_lower for marker in NOT_FOUND_MARKERS):
        return {"accept": False, "reason": "not_found_should_retry_web"}

    if not has_sources and len(answer_lower) < 20:
        return {"accept": False, "reason": "too_short_no_sources"}

    return {"accept": True, "reason": "ok"}


# ── LLM-Based Reflection ────────────────────────────────────────────────────

def llm_reflect(query: str, answer: str, complexity_level: str, min_words: int) -> dict:
    """
    LLM-based answer quality check with complexity-aware criteria.

    Returns:
        {"good": bool, "reason": str, "retry_source": str, "complexity": str}
    """
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
        raw = call_llm(prompt, max_tokens=128)
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            reflection = json.loads(match.group())
            reflection["complexity"] = complexity_level
            logger.info(f"🔍 LLM Reflection [{complexity_level}]: {reflection}")
            return reflection
    except Exception as e:
        logger.warning(f"⚠️ LLM reflection failed: {e}")

    return {"good": True, "reason": "default pass", "retry_source": "web", "complexity": complexity_level}


# ── Reflector Class ──────────────────────────────────────────────────────────

class Reflector:
    """
    Unified reflection engine.
    Combines lightweight and LLM-based reflection.
    """

    def reflect(
        self,
        query: str,
        answer: str,
        sources: list = None,
        complexity: dict = None,
        use_llm: bool = True,
    ) -> dict:
        """
        Run reflection on an answer.

        Args:
            query: Original user query
            answer: Generated answer
            sources: List of source names
            complexity: Complexity info from planner
            use_llm: Whether to use LLM-based reflection (True) or fast heuristic only

        Returns:
            {"good": bool, "reason": str, "retry_source": str, "complexity": str}
        """
        sources = sources or []
        complexity = complexity or {"level": "simple", "min_words": 5}

        # Step 1: Quick heuristic check
        critique = quick_critique(query, answer, has_sources=bool(sources))

        if not critique["accept"]:
            logger.info(f"🔍 Quick critique rejected: {critique['reason']}")
            return {
                "good": False,
                "reason": critique["reason"],
                "retry_source": "web" if "not_found" in critique["reason"] else "both",
                "complexity": complexity.get("level", "simple"),
            }

        # Step 2: LLM reflection for non-trivial answers
        if use_llm and complexity.get("level") in ("moderate", "complex"):
            return llm_reflect(
                query, answer,
                complexity.get("level", "simple"),
                complexity.get("min_words", 5),
            )

        return {
            "good": True,
            "reason": "passed heuristic check",
            "retry_source": "web",
            "complexity": complexity.get("level", "simple"),
        }

    def reflect_and_retry(
        self,
        query: str,
        answer: str,
        sources: list,
        retry_fn: Callable[[str], dict],
        max_retries: int = 1,
        complexity: dict = None,
    ) -> dict:
        """
        Reflection loop with retry capability.
        If answer is rejected, calls retry_fn to get a new answer.

        Args:
            query: User query
            answer: First-attempt answer
            sources: First-attempt sources
            retry_fn: Callable(query) -> response_dict
            max_retries: Max additional attempts
            complexity: Complexity info from planner

        Returns:
            Final response dict with reflected/reflection_reason fields
        """
        reflection = self.reflect(query, answer, sources, complexity)
        logger.info(f"🔎 Reflection: {reflection}")

        if reflection.get("good") or max_retries <= 0:
            return {
                "answer": answer,
                "sources": sources,
                "reflected": False,
                "reflection_reason": reflection.get("reason", "ok"),
                "reflection": reflection,
            }

        # Retry
        logger.info(f"🔁 Reflection triggered retry ({reflection.get('reason')})")

        retry_result = retry_fn(query)

        # Recursive with one fewer retry
        nested = self.reflect_and_retry(
            query=query,
            answer=retry_result.get("answer", ""),
            sources=retry_result.get("sources", []),
            retry_fn=retry_fn,
            max_retries=max_retries - 1,
            complexity=complexity,
        )

        return {
            **{k: v for k, v in retry_result.items() if k not in ("answer", "sources")},
            **nested,
            "reflected": True,
            "reflection_reason": reflection.get("reason", "retry"),
        }


# ── Top-level Helper Functions ───────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class ReflectionResult:
    is_valid: bool
    feedback: str
    suggested_action: str  # accept | retry | replan


def reflect_on_answer(query: str, answer: str, sources: list = None, complexity: str = "simple") -> ReflectionResult:
    reflector = Reflector()
    sources = sources or []
    ans_lower = (answer or "").lower()

    if "tool execution failed" in ans_lower or "tool '" in ans_lower and "failed:" in ans_lower:
        return ReflectionResult(
            is_valid=False,
            feedback="Tool execution returned error",
            suggested_action="retry"
        )

    res = reflector.reflect(query, answer, sources, {"level": complexity})
    is_good = res.get("good", True)
    reason = res.get("reason", "ok")
    suggested_action = "accept" if is_good else ("replan" if "not_found" in reason else "retry")
    return ReflectionResult(
        is_valid=is_good,
        feedback=reason,
        suggested_action=suggested_action
    )
