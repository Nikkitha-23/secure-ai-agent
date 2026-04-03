"""
latency_tracker.py — Secure AI Agent
Observability module for tracking per-query latency metrics.

Tracks:
- Retrieval time
- Reranking time
- Time to First Token (TTFT)
- Total response time
- Per-query breakdown log → metrics/latency.jsonl
"""

import time
import json
import os
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

METRICS_DIR  = "metrics"
LATENCY_LOG  = os.path.join(METRICS_DIR, "latency.jsonl")

os.makedirs(METRICS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LATENCY TRACKER CLASS
# ─────────────────────────────────────────────

class LatencyTracker:
    """
    Tracks latency for each stage of the RAG pipeline.

    Usage:
        tracker = LatencyTracker(query="What is AI?", user_id="u001", role="student")
        tracker.start("retrieval")
        # ... do retrieval ...
        tracker.stop("retrieval")
        tracker.start("reranking")
        # ... do reranking ...
        tracker.stop("reranking")
        tracker.mark_first_token()
        tracker.finish()
        tracker.log()
    """

    STAGES = ["retrieval", "reranking", "generation", "total"]

    def __init__(self, query: str, user_id: str = "unknown", role: str = "unknown", domain: str = "unknown"):
        self.query      = query
        self.user_id    = user_id
        self.role       = role
        self.domain     = domain
        self.timestamp  = datetime.now(timezone.utc).isoformat()

        self._start_times   = {}
        self._durations     = {}
        self._first_token_t = None
        self._total_start   = time.perf_counter()

    # ── Stage Timers ──────────────────────────

    def start(self, stage: str):
        """Start timing a stage."""
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        """Stop timing a stage. Returns duration in ms."""
        if stage not in self._start_times:
            return 0.0
        duration_ms = (time.perf_counter() - self._start_times[stage]) * 1000
        self._durations[stage] = round(duration_ms, 2)
        return duration_ms

    def mark_first_token(self):
        """Mark the time to first token (TTFT)."""
        self._first_token_t = round(
            (time.perf_counter() - self._total_start) * 1000, 2
        )

    def finish(self):
        """Mark total end time."""
        self._durations["total"] = round(
            (time.perf_counter() - self._total_start) * 1000, 2
        )

    # ── Build Report ──────────────────────────

    def get_report(self) -> dict:
        """Return full latency report as dict."""
        return {
            "timestamp":          self.timestamp,
            "query":              self.query[:100],  # truncate long queries
            "user_id":            self.user_id,
            "role":               self.role,
            "domain":             self.domain,
            "latency_ms": {
                "retrieval":      self._durations.get("retrieval", None),
                "reranking":      self._durations.get("reranking", None),
                "generation":     self._durations.get("generation", None),
                "total":          self._durations.get("total", None),
                "ttft":           self._first_token_t,
            },
            "performance_grade":  self._grade(),
        }

    def _grade(self) -> str:
        """Grade performance based on total latency."""
        total = self._durations.get("total", None)
        if total is None:
            return "unknown"
        if total < 500:
            return "excellent"   # < 500ms
        elif total < 1500:
            return "good"        # < 1.5s
        elif total < 3000:
            return "acceptable"  # < 3s
        else:
            return "slow"        # > 3s

    # ── Logging ───────────────────────────────

    def log(self):
        """Append latency report to metrics/latency.jsonl."""
        report = self.get_report()
        with open(LATENCY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")
        return report

    def print_summary(self):
        """Print a human-readable summary."""
        report = self.get_report()
        lat    = report["latency_ms"]
        print("\n" + "─" * 50)
        print(f"⏱️  Latency Report")
        print(f"─" * 50)
        print(f"  Query      : {report['query'][:60]}...")
        print(f"  Role       : {report['role']} ({report['domain']})")
        print(f"  Retrieval  : {lat['retrieval']} ms")
        print(f"  Reranking  : {lat['reranking']} ms")
        print(f"  Generation : {lat['generation']} ms")
        print(f"  TTFT       : {lat['ttft']} ms")
        print(f"  Total      : {lat['total']} ms")
        print(f"  Grade      : {report['performance_grade'].upper()}")
        print("─" * 50 + "\n")


# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────

def get_latency_summary() -> dict:
    """Read latency.jsonl and compute aggregate stats."""
    if not os.path.exists(LATENCY_LOG):
        return {"error": "No latency log found"}

    records = []
    with open(LATENCY_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if not records:
        return {"error": "Empty latency log"}

    def avg(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 2) if values else None

    totals     = [r["latency_ms"]["total"]     for r in records]
    retrievals = [r["latency_ms"]["retrieval"] for r in records]
    ttfts      = [r["latency_ms"]["ttft"]      for r in records]

    grades = {}
    for r in records:
        g = r.get("performance_grade", "unknown")
        grades[g] = grades.get(g, 0) + 1

    return {
        "total_queries":       len(records),
        "avg_total_ms":        avg(totals),
        "avg_retrieval_ms":    avg(retrievals),
        "avg_ttft_ms":         avg(ttfts),
        "min_total_ms":        round(min(totals), 2) if totals else None,
        "max_total_ms":        round(max(totals), 2) if totals else None,
        "performance_grades":  grades,
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Latency Tracker — Self Test")
    print("=" * 50)

    # Simulate a RAG pipeline
    tracker = LatencyTracker(
        query="What are the fee structures at Anna University?",
        user_id="u001",
        role="student",
        domain="education"
    )

    # Simulate retrieval
    tracker.start("retrieval")
    time.sleep(0.12)  # simulate 120ms retrieval
    tracker.stop("retrieval")

    # Simulate reranking
    tracker.start("reranking")
    time.sleep(0.05)  # simulate 50ms reranking
    tracker.stop("reranking")

    # Simulate generation + TTFT
    tracker.start("generation")
    time.sleep(0.08)  # simulate 80ms to first token
    tracker.mark_first_token()
    time.sleep(0.30)  # simulate rest of generation
    tracker.stop("generation")

    tracker.finish()
    tracker.print_summary()
    tracker.log()

    print("📊 Aggregate Stats:")
    stats = get_latency_summary()
    for k, v in stats.items():
        print(f"  {k}: {v}")