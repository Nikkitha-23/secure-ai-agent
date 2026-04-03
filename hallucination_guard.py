"""
hallucination_guard.py — Secure AI Agent
Grounding + Citation Enforcement + Hallucination Detection

Features:
  - Grounding score: checks how much of the answer is in retrieved chunks
  - Citation enforcement: attaches source references to answer
  - Hallucination flag: marks answer as grounded / partial / hallucinated
  - No LLM needed — pure text overlap logic
"""

import re
import json
import os
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Thresholds
GROUNDED_THRESHOLD   = 0.6   # 60%+ overlap → grounded
PARTIAL_THRESHOLD    = 0.4   # 40-60%       → partial
# below 40%                  → hallucinated

METRICS_DIR = "metrics"
HALLUCINATION_LOG = os.path.join(METRICS_DIR, "hallucination.jsonl")
os.makedirs(METRICS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────

def normalize(text: str) -> set:
    """
    Normalize text → lowercase word set.
    Removes punctuation and stopwords for fair overlap comparison.
    """
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "this", "that", "these", "those", "it", "its", "and", "or",
        "but", "not", "no", "so", "if", "as", "up", "out", "about",
    }
    text  = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text)  # only words 3+ chars
    return set(words) - STOPWORDS


def word_overlap(text_a: str, text_b: str) -> float:
    """
    Compute word overlap ratio between two texts.
    Returns 0.0 to 1.0
    """
    set_a = normalize(text_a)
    set_b = normalize(text_b)

    if not set_a:
        return 0.0

    intersection = set_a & set_b
    return len(intersection) / len(set_a)


# ─────────────────────────────────────────────
# GROUNDING CHECKER
# ─────────────────────────────────────────────

def compute_grounding_score(answer: str, chunks: list[str]) -> dict:
    """
    Check how much of the answer is grounded in retrieved chunks.

    Args:
        answer : generated answer text
        chunks : list of retrieved context chunks

    Returns:
        dict with score, label, details
    """
    if not chunks or not answer.strip():
        return {
            "score":  0.0,
            "label":  "hallucinated",
            "detail": "No chunks or empty answer",
        }

    # Combine all chunks into one reference text
    combined_chunks = " ".join(chunks)

    # Compute overlap
    score = word_overlap(answer, combined_chunks)
    score = round(score, 3)

    # Label
    if score >= GROUNDED_THRESHOLD:
        label = "grounded"
    elif score >= PARTIAL_THRESHOLD:
        label = "partial"
    else:
        label = "hallucinated"

    return {
        "score":  score,
        "label":  label,
        "detail": f"{round(score * 100, 1)}% of answer words found in retrieved chunks",
    }


# ─────────────────────────────────────────────
# CITATION ENFORCER
# ─────────────────────────────────────────────

def attach_citations(answer: str, sources: list[str], chunks: list[str]) -> str:
    """
    Attach source citations to the answer.

    Finds which chunks are most relevant to the answer
    and appends citation block.

    Args:
        answer  : generated answer
        sources : list of source file names
        chunks  : list of retrieved chunks

    Returns:
        answer with citations appended
    """
    if not sources:
        return answer

    # Find which sources contributed most
    cited_sources = []
    for i, chunk in enumerate(chunks):
        overlap = word_overlap(answer, chunk)
        if overlap > 0.2 and i < len(sources):  # 20%+ overlap → cite it
            source = sources[i] if i < len(sources) else sources[-1]
            if source not in cited_sources:
                cited_sources.append(source)

    # Fallback — cite all sources if none matched
    if not cited_sources:
        cited_sources = sources[:2]  # top 2

    # Build citation block
    citation_lines = "\n".join([f"  [{i+1}] {src}" for i, src in enumerate(cited_sources)])
    citation_block = f"\n\n📚 Sources:\n{citation_lines}"

    return answer + citation_block


# ─────────────────────────────────────────────
# MAIN GUARD FUNCTION
# ─────────────────────────────────────────────

def check_hallucination(
    query:   str,
    answer:  str,
    chunks:  list[str],
    sources: list[str],
    user_id: str = "unknown",
    role:    str = "unknown",
) -> dict:
    """
    Full hallucination check pipeline.

    1. Compute grounding score
    2. Attach citations
    3. Flag if hallucinated
    4. Log result

    Args:
        query   : original user query
        answer  : generated answer
        chunks  : retrieved context chunks
        sources : source file names
        user_id : for logging
        role    : for logging

    Returns:
        dict with:
            - grounded_answer  : answer + citations
            - grounding        : score, label, detail
            - is_safe          : True if grounded or partial
            - warning          : warning message if hallucinated
    """
    # Step 1: Grounding score
    grounding = compute_grounding_score(answer, chunks)

    # Step 2: Attach citations
    grounded_answer = attach_citations(answer, sources, chunks)

    # Step 3: Safety flag
    is_safe = grounding["label"] in ["grounded", "partial"]

    # Step 4: Warning for hallucinated answers
    warning = None
    if grounding["label"] == "hallucinated":
        warning = (
            "⚠️ WARNING: This answer could not be fully verified "
            "against the source documents. Please treat with caution."
        )
        grounded_answer = f"{grounded_answer}\n\n{warning}"

    # Step 5: Log
    log_entry = {
        "query":          query[:100],
        "user_id":        user_id,
        "role":           role,
        "grounding_score": grounding["score"],
        "grounding_label": grounding["label"],
        "sources":        sources,
        "is_safe":        is_safe,
    }
    with open(HALLUCINATION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "grounded_answer": grounded_answer,
        "grounding":       grounding,
        "is_safe":         is_safe,
        "warning":         warning,
    }


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────

def get_hallucination_stats() -> dict:
    """Read hallucination.jsonl and compute stats."""
    if not os.path.exists(HALLUCINATION_LOG):
        return {"error": "No hallucination log found"}

    records = []
    with open(HALLUCINATION_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if not records:
        return {"error": "Empty log"}

    labels = {}
    scores = []
    for r in records:
        l = r.get("grounding_label", "unknown")
        labels[l] = labels.get(l, 0) + 1
        scores.append(r.get("grounding_score", 0))

    return {
        "total_queries":      len(records),
        "avg_grounding_score": round(sum(scores) / len(scores), 3),
        "label_distribution": labels,
        "safe_rate":          round(
            (labels.get("grounded", 0) + labels.get("partial", 0)) / len(records) * 100, 1
        ),
    }


# ─────────────────────────────────────────────
# SELF TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Hallucination Guard — Self Test")
    print("=" * 55)

    # Test chunks (simulated retrieved docs)
    test_chunks = [
        "Anna University follows a semester system with 75% attendance mandatory for all students. Students below 75% are not allowed to write exams.",
        "The fee structure for B.Tech programs at Anna University includes tuition fee of Rs. 50,000 per year for government quota seats.",
        "Academic regulations state that students must complete all internal assessments and submit records before final exams.",
    ]
    test_sources = ["CEG_UG_Fee_Structure.pdf", "B.Tech.AIDS.pdf", "ACADEMIC_REGULATIONS.pdf"]

    # Test 1: Well-grounded answer
    print("\n[Test 1] Grounded Answer:")
    answer1 = "Students must maintain 75% attendance. Those below 75% attendance are not permitted to appear for exams as per university regulations."
    result1 = check_hallucination(
        query="What is the attendance policy?",
        answer=answer1,
        chunks=test_chunks,
        sources=test_sources,
        user_id="u001",
        role="student"
    )
    print(f"  Grounding : {result1['grounding']['label']} ({result1['grounding']['score']})")
    print(f"  Safe      : {result1['is_safe']}")
    print(f"  Answer    :\n{result1['grounded_answer']}")

    # Test 2: Hallucinated answer
    print("\n[Test 2] Hallucinated Answer:")
    answer2 = "Students can skip all classes and still pass exams by paying a special fee to the professor directly."
    result2 = check_hallucination(
        query="Can I skip classes?",
        answer=answer2,
        chunks=test_chunks,
        sources=test_sources,
        user_id="u002",
        role="student"
    )
    print(f"  Grounding : {result2['grounding']['label']} ({result2['grounding']['score']})")
    print(f"  Safe      : {result2['is_safe']}")
    print(f"  Warning   : {result2['warning']}")

    # Test 3: Partial answer
    print("\n[Test 3] Partial Answer:")
    answer3 = "The university has fee structures and students must follow academic regulations for exams."
    result3 = check_hallucination(
        query="Tell me about fees and exams",
        answer=answer3,
        chunks=test_chunks,
        sources=test_sources,
        user_id="u003",
        role="teacher"
    )
    print(f"  Grounding : {result3['grounding']['label']} ({result3['grounding']['score']})")
    print(f"  Safe      : {result3['is_safe']}")

    # Stats
    print("\n[Stats]")
    stats = get_hallucination_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 55)
    print("  All tests complete!")
    print("=" * 55)