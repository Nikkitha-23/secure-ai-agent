"""
rag/cache.py — Query Cache Layer
----------------------------------
Same question மறுபடியும் கேட்டா cached answer return பண்ணும்.

Two levels:
    L1 → In-memory (fastest, resets on restart)
    L2 → Disk cache JSON (persists across restarts)
"""

import json
import os
import hashlib
import logging
import time
from typing import Optional

CACHE_FILE = "query_cache.json"
CACHE_TTL = 3600   # 1 hour (seconds)
MAX_CACHE = 200    # max entries

logging.basicConfig(level=logging.INFO)


class QueryCache:

    def __init__(self):
        self._memory: dict = {}    # L1 in-memory
        self._disk: dict = {}      # L2 disk
        self._load()

    def _load(self):
        """Load disk cache."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    self._disk = json.load(f)
                logging.info(f"✅ Cache loaded: {len(self._disk)} entries")
            except:
                self._disk = {}

    def _save(self):
        """Save disk cache."""
        try:
            # Keep only MAX_CACHE entries
            if len(self._disk) > MAX_CACHE:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._disk.keys(),
                    key=lambda k: self._disk[k].get("timestamp", 0)
                )
                for key in sorted_keys[:len(self._disk) - MAX_CACHE]:
                    del self._disk[key]

            with open(CACHE_FILE, "w") as f:
                json.dump(self._disk, f, indent=2)
        except Exception as e:
            logging.error(f"❌ Cache save failed: {e}")

    def _hash(self, question: str) -> str:
        """Create cache key from question."""
        normalized = question.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, question: str) -> Optional[dict]:
        """
        Get cached answer for question.
        Returns None if not found or expired.
        """
        key = self._hash(question)
        now = time.time()

        # L1: Check memory first
        if key in self._memory:
            entry = self._memory[key]
            if now - entry["timestamp"] < CACHE_TTL:
                logging.info(f"⚡ L1 Cache HIT: '{question[:50]}'")
                return entry["data"]
            else:
                del self._memory[key]

        # L2: Check disk
        if key in self._disk:
            entry = self._disk[key]
            if now - entry["timestamp"] < CACHE_TTL:
                # Promote to L1
                self._memory[key] = entry
                logging.info(f"💾 L2 Cache HIT: '{question[:50]}'")
                return entry["data"]
            else:
                del self._disk[key]

        logging.info(f"❌ Cache MISS: '{question[:50]}'")
        return None

    def set(self, question: str, data: dict):
        """Save answer to cache."""
        key = self._hash(question)
        entry = {
            "question": question,
            "timestamp": time.time(),
            "data": data
        }
        # Save to both L1 and L2
        self._memory[key] = entry
        self._disk[key] = entry
        self._save()
        logging.info(f"💾 Cache SET: '{question[:50]}'")

    def clear(self):
        """Clear all cache."""
        self._memory = {}
        self._disk = {}
        self._save()
        logging.info("🗑️ Cache cleared")

    def stats(self) -> dict:
        """Return cache statistics."""
        now = time.time()
        valid = sum(
            1 for e in self._disk.values()
            if now - e.get("timestamp", 0) < CACHE_TTL
        )
        return {
            "memory_entries": len(self._memory),
            "disk_entries": len(self._disk),
            "valid_entries": valid,
            "ttl_seconds": CACHE_TTL,
            "max_entries": MAX_CACHE
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
cache = QueryCache()


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test cache
    test_data = {"answer": "AI is artificial intelligence", "sources": [], "search_type": "pdf"}

    print("🔍 First request (MISS):")
    result = cache.get("What is AI?")
    print(f"  Result: {result}")

    print("\n💾 Setting cache...")
    cache.set("What is AI?", test_data)

    print("\n⚡ Second request (HIT):")
    result = cache.get("What is AI?")
    print(f"  Result: {result}")

    print("\n⚡ Case insensitive test:")
    result = cache.get("what is ai?")
    print(f"  Result: {result}")

    print(f"\n📊 Stats: {cache.stats()}")
