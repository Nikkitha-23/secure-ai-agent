"""
rag/monitor.py — Cost + Latency Monitor
----------------------------------------
ஒவ்வொரு query-க்கும் track பண்ணும்:
→ Latency (seconds)
→ Token count (estimated)
→ Search type
→ Timestamp

Usage:
    from rag.monitor import monitor
    monitor.start("query_id")
    monitor.end("query_id", tokens=100, search_type="web")
    monitor.summary()
"""

import time
import json
import os
import logging
from datetime import datetime
from typing import Optional

MONITOR_LOG = "monitor_log.json"
logging.basicConfig(level=logging.INFO)


class Monitor:
    def __init__(self):
        self._active: dict = {}   # query_id → start time
        self._log: list = []      # all records
        self._load()

    def _load(self):
        """Load existing log from file."""
        if os.path.exists(MONITOR_LOG):
            try:
                with open(MONITOR_LOG, "r") as f:
                    self._log = json.load(f)
            except:
                self._log = []

    def _save(self):
        """Save log to file."""
        try:
            with open(MONITOR_LOG, "w") as f:
                json.dump(self._log[-500:], f, indent=2)  # keep last 500
        except Exception as e:
            logging.error(f"❌ Monitor save failed: {e}")

    def start(self, query_id: str):
        """Start timing a query."""
        self._active[query_id] = time.time()

    def end(
        self,
        query_id: str,
        question: str = "",
        search_type: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        status: str = "success"
    ):
        """End timing and record metrics."""
        start_time = self._active.pop(query_id, None)
        if start_time is None:
            return

        latency = round(time.time() - start_time, 3)
        total_tokens = input_tokens + output_tokens

        # Groq pricing (approx): llama-3.1-8b-instant
        # $0.05 per 1M input tokens, $0.08 per 1M output tokens
        cost_usd = round(
            (input_tokens * 0.05 / 1_000_000) +
            (output_tokens * 0.08 / 1_000_000),
            8
        )

        record = {
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "question": question[:80],
            "latency_sec": latency,
            "search_type": search_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "status": status
        }

        self._log.append(record)
        self._save()

        logging.info(
            f"📊 Monitor | latency={latency}s | tokens={total_tokens} | "
            f"cost=${cost_usd:.6f} | search={search_type}"
        )
        return record

    def summary(self) -> dict:
        """Return summary statistics."""
        if not self._log:
            return {"message": "No data yet"}

        latencies = [r["latency_sec"] for r in self._log]
        tokens = [r["total_tokens"] for r in self._log]
        costs = [r["cost_usd"] for r in self._log]

        search_counts = {}
        for r in self._log:
            s = r["search_type"]
            search_counts[s] = search_counts.get(s, 0) + 1

        return {
            "total_queries": len(self._log),
            "avg_latency_sec": round(sum(latencies) / len(latencies), 3),
            "max_latency_sec": max(latencies),
            "min_latency_sec": min(latencies),
            "total_tokens": sum(tokens),
            "avg_tokens_per_query": round(sum(tokens) / len(tokens)),
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_per_query_usd": round(sum(costs) / len(costs), 6),
            "search_type_breakdown": search_counts,
            "last_10": self._log[-10:]
        }

    def recent(self, n: int = 10) -> list:
        """Return last n records."""
        return self._log[-n:]

    def clear(self):
        """Clear all logs."""
        self._log = []
        self._save()
        logging.info("🗑️ Monitor log cleared")


# ── Singleton ──────────────────────────────────────────────────────────────────
monitor = Monitor()


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uuid

    for i in range(3):
        qid = str(uuid.uuid4())
        monitor.start(qid)
        time.sleep(0.5)
        monitor.end(
            qid,
            question=f"Test question {i+1}",
            search_type=["pdf", "web", "both"][i],
            input_tokens=200 + i * 50,
            output_tokens=100 + i * 30
        )

    print("\n📊 Monitor Summary:")
    summary = monitor.summary()
    for k, v in summary.items():
        if k != "last_10":
            print(f"  {k}: {v}")
