"""
retrieval_benchmark.py
======================
Benchmarks the hybrid retrieval pipeline (ChromaDB + BM25) for the
Secure Multi-Tenant Document Intelligence Agent (Education & Healthcare).

Metrics evaluated:
  - Precision@K, Recall@K, MRR (Mean Reciprocal Rank)
  - Hit Rate@K
  - Average Retrieval Latency
  - Tenant Isolation Integrity (cross-tenant leakage check)

Usage:
    python retrieval_benchmark.py
    python retrieval_benchmark.py --tenant anna_univ --top_k 5
"""

import time
import argparse
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkQuery:
    query_id: str
    query_text: str
    tenant_id: str
    domain: str                        # "education" | "healthcare"
    relevant_doc_ids: List[str]        # Ground-truth relevant document IDs
    role: str = "student"              # student | teacher | doctor | nurse


@dataclass
class RetrievalResult:
    query_id: str
    retrieved_doc_ids: List[str]
    latency_ms: float
    scores: List[float] = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    query_id: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: bool
    latency_ms: float
    tenant_isolation_pass: bool


# ---------------------------------------------------------------------------
# Ground-Truth Dataset (Education + Healthcare)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: List[BenchmarkQuery] = [

    # ── Education: Anna University ──────────────────────────────────────────
    BenchmarkQuery(
        query_id="edu_001",
        query_text="What are the attendance requirements for Anna University students?",
        tenant_id="anna_univ",
        domain="education",
        relevant_doc_ids=["anna_policy_attendance", "anna_regulation_2021"],
        role="student",
    ),
    BenchmarkQuery(
        query_id="edu_002",
        query_text="Explain the syllabus for Data Structures and Algorithms.",
        tenant_id="anna_univ",
        domain="education",
        relevant_doc_ids=["anna_syllabus_cse_sem3", "anna_syllabus_dsa"],
        role="student",
    ),
    BenchmarkQuery(
        query_id="edu_003",
        query_text="What is the grading scheme for internal assessments?",
        tenant_id="anna_univ",
        domain="education",
        relevant_doc_ids=["anna_regulation_2021", "anna_assessment_policy"],
        role="teacher",
    ),

    # ── Education: VIT ───────────────────────────────────────────────────────
    BenchmarkQuery(
        query_id="edu_004",
        query_text="How does VIT calculate CGPA for semester results?",
        tenant_id="vit",
        domain="education",
        relevant_doc_ids=["vit_academic_regulation", "vit_grading_policy"],
        role="student",
    ),
    BenchmarkQuery(
        query_id="edu_005",
        query_text="What is the CAT (Continuous Assessment Test) schedule at VIT?",
        tenant_id="vit",
        domain="education",
        relevant_doc_ids=["vit_cat_schedule_2024", "vit_academic_calendar"],
        role="student",
    ),

    # ── Healthcare: Hospital A ───────────────────────────────────────────────
    BenchmarkQuery(
        query_id="hc_001",
        query_text="What is the standard dosage protocol for metformin in Type 2 diabetes?",
        tenant_id="hospital_a",
        domain="healthcare",
        relevant_doc_ids=["hca_drug_protocol_metformin", "hca_diabetes_guidelines"],
        role="doctor",
    ),
    BenchmarkQuery(
        query_id="hc_002",
        query_text="What are the post-operative care instructions after appendectomy?",
        tenant_id="hospital_a",
        domain="healthcare",
        relevant_doc_ids=["hca_surgical_sop_appendectomy", "hca_postop_care"],
        role="nurse",
    ),
    BenchmarkQuery(
        query_id="hc_003",
        query_text="What is the patient discharge checklist for ICU patients?",
        tenant_id="hospital_a",
        domain="healthcare",
        relevant_doc_ids=["hca_icu_discharge_protocol", "hca_discharge_checklist"],
        role="doctor",
    ),
]

# Cross-tenant isolation test pairs: (query_tenant, forbidden_tenant, query_id)
ISOLATION_TESTS: List[Tuple[str, str, str]] = [
    ("anna_univ", "vit",        "edu_001"),   # Anna Univ query must NOT return VIT docs
    ("vit",       "anna_univ",  "edu_004"),   # VIT query must NOT return Anna Univ docs
    ("hospital_a","anna_univ",  "hc_001"),    # Healthcare must NOT bleed into Education
]


# ---------------------------------------------------------------------------
# Mock Retriever (replace with your actual retriever import)
# ---------------------------------------------------------------------------

def _mock_retrieve(query: str, tenant_id: str, top_k: int) -> Tuple[List[str], List[float], float]:
    """
    MOCK implementation — replace with your actual hybrid retrieval call.

    Expected real call example:
        from rag_pipeline import hybrid_retrieve
        results = hybrid_retrieve(query=query, tenant_id=tenant_id, top_k=top_k)
        doc_ids  = [r["doc_id"] for r in results]
        scores   = [r["score"]  for r in results]

    Returns:
        (doc_ids, scores, latency_ms)
    """
    import random
    import hashlib

    # Simulate retrieval: seed with query so output is deterministic per query
    seed = int(hashlib.md5(f"{query}{tenant_id}".encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)

    # Pool of fake doc IDs including some that belong to the correct tenant
    tenant_pool = {
        "anna_univ":  ["anna_policy_attendance", "anna_regulation_2021",
                        "anna_syllabus_cse_sem3", "anna_assessment_policy"],
        "vit":        ["vit_academic_regulation", "vit_grading_policy",
                        "vit_cat_schedule_2024",  "vit_academic_calendar"],
        "hospital_a": ["hca_drug_protocol_metformin", "hca_diabetes_guidelines",
                        "hca_surgical_sop_appendectomy", "hca_postop_care",
                        "hca_icu_discharge_protocol", "hca_discharge_checklist"],
    }
    pool   = tenant_pool.get(tenant_id, []) + ["generic_doc_01", "generic_doc_02"]
    chosen = rng.sample(pool, min(top_k, len(pool)))
    scores = sorted([round(rng.uniform(0.5, 1.0), 4) for _ in chosen], reverse=True)

    latency_ms = round(rng.uniform(30, 200), 2)   # Simulated latency
    return chosen, scores, latency_ms


# ---------------------------------------------------------------------------
# Metric Calculators
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int) -> bool:
    return any(doc in relevant for doc in retrieved[:k])


# ---------------------------------------------------------------------------
# Isolation Checker
# ---------------------------------------------------------------------------

def check_tenant_isolation(
    query_text: str,
    correct_tenant: str,
    forbidden_tenant: str,
    top_k: int,
) -> bool:
    """
    Returns True if NO document from the forbidden tenant appears in results.
    """
    forbidden_prefixes = {
        "anna_univ":  "anna_",
        "vit":        "vit_",
        "hospital_a": "hca_",
    }
    prefix = forbidden_prefixes.get(forbidden_tenant, forbidden_tenant + "_")

    doc_ids, _, _ = _mock_retrieve(query_text, correct_tenant, top_k)
    leaked = [d for d in doc_ids if d.startswith(prefix)]
    return len(leaked) == 0


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    queries: Optional[List[BenchmarkQuery]] = None,
    top_k: int = 5,
    tenant_filter: Optional[str] = None,
) -> Dict:

    queries = queries or BENCHMARK_QUERIES
    if tenant_filter:
        queries = [q for q in queries if q.tenant_id == tenant_filter]

    results: List[BenchmarkMetrics] = []

    print("\n" + "=" * 70)
    print("  RETRIEVAL BENCHMARK — Secure AI Agent")
    print(f"  Queries: {len(queries)}  |  Top-K: {top_k}  |  Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    for q in queries:
        doc_ids, scores, latency_ms = _mock_retrieve(q.query_text, q.tenant_id, top_k)

        p_at_k  = precision_at_k(doc_ids, q.relevant_doc_ids, top_k)
        r_at_k  = recall_at_k(doc_ids, q.relevant_doc_ids, top_k)
        mrr     = mean_reciprocal_rank(doc_ids, q.relevant_doc_ids)
        hit     = hit_rate_at_k(doc_ids, q.relevant_doc_ids, top_k)

        # Quick isolation check: retrieved docs should not belong to other tenants
        all_tenants = {bq.tenant_id for bq in BENCHMARK_QUERIES}
        other_tenants = all_tenants - {q.tenant_id}
        isolation_ok = all(
            not doc_id.startswith(t.split("_")[0]) or q.tenant_id == t
            for doc_id in doc_ids
            for t in other_tenants
        )

        m = BenchmarkMetrics(
            query_id=q.query_id,
            precision_at_k=round(p_at_k, 4),
            recall_at_k=round(r_at_k, 4),
            mrr=round(mrr, 4),
            hit_rate=hit,
            latency_ms=latency_ms,
            tenant_isolation_pass=isolation_ok,
        )
        results.append(m)

        status = "✅" if hit else "❌"
        print(f"  [{status}] {q.query_id:<10}  P@{top_k}={p_at_k:.2f}  R@{top_k}={r_at_k:.2f}  "
              f"MRR={mrr:.2f}  Latency={latency_ms:.1f}ms  "
              f"Isolation={'✅' if isolation_ok else '🚨'}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    n = len(results)
    avg_p   = sum(r.precision_at_k for r in results) / n
    avg_r   = sum(r.recall_at_k    for r in results) / n
    avg_mrr = sum(r.mrr            for r in results) / n
    avg_lat = sum(r.latency_ms     for r in results) / n
    hit_pct = sum(r.hit_rate       for r in results) / n * 100
    iso_pct = sum(r.tenant_isolation_pass for r in results) / n * 100

    print("\n" + "-" * 70)
    print("  AGGREGATE RESULTS")
    print(f"  Avg Precision@{top_k}   : {avg_p:.4f}")
    print(f"  Avg Recall@{top_k}      : {avg_r:.4f}")
    print(f"  Mean Reciprocal Rank : {avg_mrr:.4f}")
    print(f"  Hit Rate@{top_k}        : {hit_pct:.1f}%")
    print(f"  Avg Latency          : {avg_lat:.1f} ms")
    print(f"  Tenant Isolation     : {iso_pct:.1f}%")

    # ── Isolation suite ──────────────────────────────────────────────────────
    print("\n  ISOLATION SUITE")
    for correct, forbidden, qid in ISOLATION_TESTS:
        q_obj = next((q for q in BENCHMARK_QUERIES if q.query_id == qid), None)
        if not q_obj:
            continue
        passed = check_tenant_isolation(q_obj.query_text, correct, forbidden, top_k)
        print(f"  {'✅' if passed else '🚨'}  {qid}  ({correct} → should NOT see {forbidden} docs)")

    print("=" * 70 + "\n")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "top_k": top_k,
        "num_queries": n,
        "avg_precision": round(avg_p, 4),
        "avg_recall": round(avg_r, 4),
        "avg_mrr": round(avg_mrr, 4),
        "hit_rate_pct": round(hit_pct, 2),
        "avg_latency_ms": round(avg_lat, 2),
        "tenant_isolation_pct": round(iso_pct, 2),
        "per_query": [asdict(r) for r in results],
    }

    # Save results
    output_file = f"benchmark_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {output_file}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval Benchmark for Secure AI Agent")
    parser.add_argument("--top_k",   type=int, default=5,    help="Top-K documents to retrieve")
    parser.add_argument("--tenant",  type=str, default=None, help="Filter by tenant ID")
    args = parser.parse_args()

    run_benchmark(top_k=args.top_k, tenant_filter=args.tenant)