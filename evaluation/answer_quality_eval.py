"""
answer_quality_eval.py — Secure AI Agent
Evaluation: Faithfulness + Factual Correctness + Robustness

Metrics:
  1. Faithfulness     — Is the answer aligned with retrieved chunks?
  2. Factual Correctness — Are key facts from docs present in answer?
  3. Robustness       — Does classifier handle noisy/typo queries correctly?

No LLM needed — pure text overlap + embedding similarity
"""

import re
import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_NAME   = "all-MiniLM-L6-v2"
METRICS_DIR  = "metrics"
EVAL_LOG     = os.path.join(METRICS_DIR, "answer_quality.jsonl")
os.makedirs(METRICS_DIR, exist_ok=True)

# Thresholds
FAITHFUL_THRESHOLD   = 0.55
FACTUAL_THRESHOLD    = 0.50
ROBUSTNESS_THRESHOLD = 0.70

# ─────────────────────────────────────────────
# MODEL (shared with domain_classifier)
# ─────────────────────────────────────────────

print("🔄 Loading embedding model for evaluation...")
_model = SentenceTransformer(MODEL_NAME)
print("✅ Evaluation model ready")

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def normalize_words(text: str) -> set:
    """Lowercase word set, remove stopwords."""
    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","to","of","in","for",
        "on","with","at","by","from","this","that","these","those",
        "it","its","and","or","but","not","no","so","if","as","up",
    }
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return set(words) - STOPWORDS


def embedding_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts."""
    emb_a = _model.encode(text_a, convert_to_tensor=True)
    emb_b = _model.encode(text_b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b)[0][0])


# ─────────────────────────────────────────────
# METRIC 1: FAITHFULNESS
# ─────────────────────────────────────────────

def evaluate_faithfulness(answer: str, chunks: list[str]) -> dict:
    """
    Faithfulness: Is the answer semantically aligned with retrieved chunks?

    Method: Embedding similarity between answer and combined chunks.
    High similarity → answer stays faithful to source.

    Returns:
        score (0-1), label, detail
    """
    if not answer.strip() or not chunks:
        return {"score": 0.0, "label": "unfaithful", "detail": "Empty input"}

    combined = " ".join(chunks[:3])  # top 3 chunks
    score    = embedding_similarity(answer, combined)
    score    = round(score, 3)

    label  = "faithful"   if score >= FAITHFUL_THRESHOLD else "unfaithful"
    detail = f"Semantic similarity to retrieved chunks: {score:.3f}"

    return {"score": score, "label": label, "detail": detail}


# ─────────────────────────────────────────────
# METRIC 2: FACTUAL CORRECTNESS
# ─────────────────────────────────────────────

def evaluate_factual_correctness(answer: str, chunks: list[str]) -> dict:
    """
    Factual Correctness: Are the key content words in the answer
    actually present in the retrieved documents?

    Method: Word overlap — answer words found in chunks.

    Returns:
        score (0-1), label, matched_facts, missing_facts
    """
    if not answer.strip() or not chunks:
        return {
            "score":         0.0,
            "label":         "incorrect",
            "matched_facts": [],
            "missing_facts": [],
            "detail":        "Empty input",
        }

    combined_chunks = " ".join(chunks)
    answer_words    = normalize_words(answer)
    chunk_words     = normalize_words(combined_chunks)

    matched = list(answer_words & chunk_words)
    missing = list(answer_words - chunk_words)

    score = round(len(matched) / len(answer_words), 3) if answer_words else 0.0
    label = "correct" if score >= FACTUAL_THRESHOLD else "incorrect"

    return {
        "score":         score,
        "label":         label,
        "matched_facts": sorted(matched)[:10],   # top 10
        "missing_facts": sorted(missing)[:10],   # top 10
        "detail":        f"{len(matched)}/{len(answer_words)} content words verified",
    }


# ─────────────────────────────────────────────
# METRIC 3: ROBUSTNESS
# ─────────────────────────────────────────────

def add_noise(query: str, noise_type: str) -> str:
    """Add noise to a query for robustness testing."""
    if noise_type == "typo":
        # Swap a character in a random word
        words = query.split()
        if len(words) > 2:
            word = words[1]
            if len(word) > 2:
                words[1] = word[0] + word[2] + word[1] + word[3:]
        return " ".join(words)

    elif noise_type == "lowercase":
        return query.lower()

    elif noise_type == "extra_spaces":
        return "  ".join(query.split())

    elif noise_type == "truncated":
        words = query.split()
        return " ".join(words[:max(2, len(words)//2)])

    return query


def evaluate_robustness(test_cases: list[dict]) -> dict:
    """
    Robustness: Does domain classifier handle noisy queries correctly?

    test_cases: list of {"query": str, "expected_domain": str}

    For each query, tests 4 noise variants:
      - original, typo, lowercase, truncated

    Returns:
        overall_score, per_noise_accuracy, details
    """
    from domain_classifier import classify_domain

    noise_types    = ["original", "typo", "lowercase", "truncated"]
    results        = []
    correct_counts = {n: 0 for n in noise_types}
    total          = len(test_cases)

    for case in test_cases:
        query    = case["query"]
        expected = case["expected_domain"]
        case_result = {"query": query, "expected": expected, "noise_results": {}}

        for noise in noise_types:
            noisy_query = query if noise == "original" else add_noise(query, noise)
            prediction  = classify_domain(noisy_query, log=False)
            correct     = prediction["domain"] == expected

            if correct:
                correct_counts[noise] += 1

            case_result["noise_results"][noise] = {
                "noisy_query": noisy_query,
                "predicted":   prediction["domain"],
                "confidence":  prediction["confidence"],
                "correct":     correct,
            }

        results.append(case_result)

    # Per-noise accuracy
    per_noise_accuracy = {
        noise: round(correct_counts[noise] / total * 100, 1)
        for noise in noise_types
    }

    # Overall robustness score
    total_correct = sum(correct_counts.values())
    total_tests   = total * len(noise_types)
    overall_score = round(total_correct / total_tests * 100, 1)

    label = "robust" if overall_score >= ROBUSTNESS_THRESHOLD * 100 else "fragile"

    return {
        "overall_score":       overall_score,
        "label":               label,
        "per_noise_accuracy":  per_noise_accuracy,
        "total_tests":         total_tests,
        "details":             results,
    }


# ─────────────────────────────────────────────
# FULL EVAL PIPELINE
# ─────────────────────────────────────────────

def run_full_eval(eval_cases: list[dict]) -> dict:
    """
    Run full evaluation pipeline on a list of cases.

    Each case:
    {
        "query":           str,
        "answer":          str,
        "chunks":          list[str],
        "sources":         list[str],
        "expected_domain": str,
    }

    Returns aggregated metrics.
    """
    faith_scores   = []
    factual_scores = []
    all_results    = []

    for case in eval_cases:
        faith   = evaluate_faithfulness(case["answer"],  case["chunks"])
        factual = evaluate_factual_correctness(case["answer"], case["chunks"])

        faith_scores.append(faith["score"])
        factual_scores.append(factual["score"])

        result = {
            "query":       case["query"][:80],
            "faithfulness":        faith,
            "factual_correctness": factual,
        }
        all_results.append(result)

        # Log
        with open(EVAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    # Robustness test
    robustness_cases = [
        {"query": c["query"], "expected_domain": c["expected_domain"]}
        for c in eval_cases
    ]
    robustness = evaluate_robustness(robustness_cases)

    avg_faith   = round(sum(faith_scores)   / len(faith_scores),   3)
    avg_factual = round(sum(factual_scores) / len(factual_scores), 3)

    summary = {
        "total_cases":              len(eval_cases),
        "avg_faithfulness":         avg_faith,
        "avg_factual_correctness":  avg_factual,
        "robustness_score":         robustness["overall_score"],
        "robustness_label":         robustness["label"],
        "robustness_per_noise":     robustness["per_noise_accuracy"],
        "overall_grade":            _grade(avg_faith, avg_factual, robustness["overall_score"]),
    }

    return {"summary": summary, "details": all_results, "robustness": robustness}


def _grade(faith: float, factual: float, robustness: float) -> str:
    avg = (faith + factual + robustness / 100) / 3
    if avg >= 0.80: return "A — Production Ready"
    if avg >= 0.65: return "B — Good, minor fixes needed"
    if avg >= 0.50: return "C — Acceptable, needs improvement"
    return           "D — Needs significant work"


# ─────────────────────────────────────────────
# SELF TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Answer Quality Evaluation — Full Test")
    print("=" * 60)

    # Sample eval cases
    eval_cases = [
        {
            "query": "What is the attendance policy?",
            "answer": "Students must maintain 75% attendance. Those below 75% are not allowed to write exams as per Anna University regulations.",
            "chunks": [
                "Anna University mandates 75% attendance for all students enrolled in B.Tech programs. Students failing to meet this requirement will be barred from examinations.",
                "Academic regulations state attendance is calculated semester-wise. Medical leave may be considered with proper documentation.",
            ],
            "sources": ["ACADEMIC_REGULATIONS.pdf"],
            "expected_domain": "education",
        },
        {
            "query": "What is the fee structure?",
            "answer": "The tuition fee for B.Tech at Anna University is Rs. 50,000 per year for government quota seats.",
            "chunks": [
                "Fee structure for B.Tech programs: Government quota - Rs. 50,000 per annum. Management quota - Rs. 1,50,000 per annum.",
                "Additional fees include examination fee, lab fee, and library fee collected each semester.",
            ],
            "sources": ["CEG_UG_Fee_Structure.pdf"],
            "expected_domain": "education",
        },
        {
            "query": "What is the dosage for paracetamol?",
            "answer": "The standard dosage for paracetamol is 500mg to 1000mg per dose, taken every 4-6 hours as needed.",
            "chunks": [
                "Paracetamol: Recommended adult dosage is 500-1000mg every 4-6 hours. Maximum daily dose should not exceed 4000mg.",
                "Paracetamol is used for mild to moderate pain relief and fever reduction. Avoid in patients with liver disease.",
            ],
            "sources": ["drug_info.pdf"],
            "expected_domain": "healthcare",
        },
        {
            "query": "How should I treat a patient with high blood pressure?",
            "answer": "Treatment for hypertension includes lifestyle changes, dietary modifications, and antihypertensive medications as prescribed by a doctor.",
            "chunks": [
                "Hypertension management: First-line treatment includes lifestyle modifications such as reduced sodium intake, regular exercise, and weight management.",
                "Pharmacological treatment for hypertension includes ACE inhibitors, calcium channel blockers, and diuretics based on patient profile.",
            ],
            "sources": ["treatment_protocols.pdf"],
            "expected_domain": "healthcare",
        },
    ]

    print("\n[Running Full Evaluation...]")
    results = run_full_eval(eval_cases)
    summary = results["summary"]

    print("\n📊 SUMMARY")
    print("─" * 40)
    print(f"  Total Cases           : {summary['total_cases']}")
    print(f"  Avg Faithfulness      : {summary['avg_faithfulness']} ({round(summary['avg_faithfulness']*100,1)}%)")
    print(f"  Avg Factual Correct   : {summary['avg_factual_correctness']} ({round(summary['avg_factual_correctness']*100,1)}%)")
    print(f"  Robustness Score      : {summary['robustness_score']}%")
    print(f"  Robustness Label      : {summary['robustness_label']}")
    print(f"  Overall Grade         : {summary['overall_grade']}")

    print("\n📊 ROBUSTNESS PER NOISE TYPE")
    print("─" * 40)
    for noise, acc in summary["robustness_per_noise"].items():
        bar = "█" * int(acc / 10)
        print(f"  {noise:12s} : {bar:10s} {acc}%")

    print("\n📊 PER CASE RESULTS")
    print("─" * 40)
    for detail in results["details"]:
        print(f"\n  Query    : {detail['query'][:55]}")
        print(f"  Faith    : {detail['faithfulness']['label']} ({detail['faithfulness']['score']})")
        print(f"  Factual  : {detail['factual_correctness']['label']} ({detail['factual_correctness']['score']})")
        print(f"  Verified : {detail['factual_correctness']['detail']}")

    print("\n" + "=" * 60)
    print("  Evaluation complete! → metrics/answer_quality.jsonl")
    print("=" * 60)