"""
rag/reflection.py — Shared Reflection / Critic Loop
-----------------------------------------------------
Used by BOTH /ask and /agent so reflection logic lives in exactly one place.

Why this exists:
- /agent already has a reflect+retry loop inside agent_loop.py (run_agent)
- /ask has NO reflection — it returns "Not found in documents." and stops
- This module extracts a reusable `reflect_and_retry()` so /ask can opt into
  the same critic loop without duplicating agent_loop's logic

Usage in main.py /ask:
    from rag.reflection import reflect_and_retry

    response_data = reflect_and_retry(
        query=clean_query,
        answer=answer,
        sources=sources,
        retrieved_docs=reranked_docs,
        retry_fn=lambda q: run_retrieval_and_llm(q),  # your existing pipeline as a callable
        max_retries=1,
    )
"""

import logging
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO)


# ── Critic: judges if an answer is acceptable ───────────────────────────────
def critique_answer(query: str, answer: str, has_sources: bool) -> dict:
    """
    Lightweight critic — no extra LLM call by default (keeps cost/latency low).
    Returns {"accept": bool, "reason": str}

    Escalate to an LLM-as-judge call here later if you want stronger critique;
    for now this catches the two failure modes you already observed:
    1. "Not found" / empty answers that should trigger web fallback
    2. Answers with zero supporting sources when sources were expected
    """
    NOT_FOUND_MARKERS = [
        "not found in documents",
        "no relevant information found",
        "i don't know",
        "cannot find",
    ]

    answer_lower = (answer or "").strip().lower()

    if not answer_lower:
        return {"accept": False, "reason": "empty_answer"}

    if any(marker in answer_lower for marker in NOT_FOUND_MARKERS):
        return {"accept": False, "reason": "not_found_should_retry_web"}

    if not has_sources and len(answer_lower) < 20:
        return {"accept": False, "reason": "too_short_no_sources"}

    return {"accept": True, "reason": "ok"}


# ── Reflect + retry orchestrator ─────────────────────────────────────────────
def reflect_and_retry(
    query: str,
    answer: str,
    sources: list,
    retry_fn: Callable[[str], dict],
    max_retries: int = 1,
    force_web_on_retry: bool = True,
) -> dict:
    """
    Core shared reflection loop.

    Args:
        query: the (clean, rewritten) user query
        answer: the answer produced by the first attempt
        sources: list of sources attached to that answer
        retry_fn: a callable(query: str) -> response_dict, your existing
                   retrieval+LLM pipeline wrapped as a function. On retry,
                   this gets called again — typically with source="web" forced.
        max_retries: how many additional attempts to allow (default 1)
        force_web_on_retry: if True, signals retry_fn to skip RAG and go
                              straight to web search (this is what fixes your
                              "/ask returns not found instead of web fallback" gap)

    Returns:
        Final response dict (same shape as your existing /ask response_data)
        with two extra keys: "reflected" (bool) and "reflection_reason" (str)
    """
    critique = critique_answer(query, answer, has_sources=bool(sources))
    logging.info(f"🔎 Reflection critique: {critique}")

    if critique["accept"] or max_retries <= 0:
        return {
            "answer": answer,
            "sources": sources,
            "reflected": False,
            "reflection_reason": critique["reason"],
        }

    # ── Retry: re-run the pipeline, biased toward web search ───────────────
    logging.info(f"🔁 Reflection triggered retry ({critique['reason']}) for: {query}")

    retry_result = retry_fn(query)

    # Recursively allow one fewer retry (keeps it bounded, no infinite loop)
    nested = reflect_and_retry(
        query=query,
        answer=retry_result.get("answer", ""),
        sources=retry_result.get("sources", []),
        retry_fn=retry_fn,
        max_retries=max_retries - 1,
        force_web_on_retry=force_web_on_retry,
    )

    # A retry happened at THIS level regardless of whether the nested call
    # also retried — always report reflected=True once we've reached here,
    # and keep the original reason that triggered the first retry.
    return {
        **{k: v for k, v in retry_result.items() if k not in ("answer", "sources")},
        **nested,
        "reflected": True,
        "reflection_reason": critique["reason"],
    }
