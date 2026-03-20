"""
rag/memory/observability.py — Memory Audit Log + Rollback
-----------------------------------------------------------
Tracks every memory operation for:
    - Debugging memory poisoning
    - Rolling back bad writes
    - Proving memory works (evaluation)
"""

import json
import os
import logging
import threading
from datetime import datetime
from typing import List, Optional

AUDIT_LOG_FILE = "memory_audit.json"
SNAPSHOT_FILE  = "memory_snapshot.json"
MAX_AUDIT_ENTRIES = 1000

logging.basicConfig(level=logging.INFO)


class MemoryAuditLog:
    """Tracks every memory write/read/prune operation."""

    def __init__(self):
        self._log: List[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(AUDIT_LOG_FILE):
            try:
                with open(AUDIT_LOG_FILE, "r") as f:
                    self._log = json.load(f)
            except:
                self._log = []

    def _save(self):
        try:
            # Keep last MAX_AUDIT_ENTRIES
            entries = self._log[-MAX_AUDIT_ENTRIES:]
            with open(AUDIT_LOG_FILE, "w") as f:
                json.dump(entries, f, indent=2)
        except Exception as e:
            logging.error(f"❌ Audit save failed: {e}")

    def log(
        self,
        operation: str,       # "write", "read", "prune", "rollback", "reject"
        memory_type: str,
        session_id: str,
        text: str,
        score: float = 0.0,
        reason: str = ""
    ):
        with self._lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "memory_type": memory_type,
                "session_id": session_id,
                "text_preview": text[:80],
                "score": round(score, 4),
                "reason": reason
            }
            self._log.append(entry)
            self._save()
            logging.debug(f"📋 Audit [{operation}] [{memory_type}] score={score:.2f}")

    def recent(self, n: int = 20, session_id: str = None) -> List[dict]:
        entries = self._log
        if session_id:
            entries = [e for e in entries if e.get("session_id") == session_id]
        return entries[-n:]

    def stats(self) -> dict:
        ops = {}
        for e in self._log:
            op = e["operation"]
            ops[op] = ops.get(op, 0) + 1
        return {
            "total_events": len(self._log),
            "by_operation": ops
        }

    def clear(self):
        with self._lock:
            self._log = []
            self._save()


class MemorySnapshot:
    """
    Periodic snapshots of memory store for rollback.
    Prevents memory poisoning from being permanent.
    """

    def __init__(self):
        self._snapshots: List[dict] = []   # {timestamp, data}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(SNAPSHOT_FILE):
            try:
                with open(SNAPSHOT_FILE, "r") as f:
                    self._snapshots = json.load(f)
            except:
                self._snapshots = []

    def _save(self):
        try:
            with open(SNAPSHOT_FILE, "w") as f:
                json.dump(self._snapshots[-5:], f, indent=2)  # keep last 5
        except Exception as e:
            logging.error(f"❌ Snapshot save failed: {e}")

    def take(self, store: list):
        """Take a snapshot of current memory store."""
        with self._lock:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "count": len(store),
                "data": [e.to_dict() for e in store]
            }
            self._snapshots.append(snapshot)
            self._save()
            logging.info(f"📸 Snapshot taken: {len(store)} memories")

    def rollback(self, store: list, index: int = -1) -> int:
        """
        Rollback memory store to a previous snapshot.
        index=-1 → most recent snapshot
        Returns number of memories restored.
        """
        with self._lock:
            if not self._snapshots:
                logging.warning("⚠️ No snapshots available for rollback")
                return 0

            from .types import MemoryEntry
            snapshot = self._snapshots[index]
            restored = [MemoryEntry.from_dict(d) for d in snapshot["data"]]
            store.clear()
            store.extend(restored)
            logging.info(f"⏪ Rollback complete: restored {len(restored)} memories from {snapshot['timestamp']}")
            return len(restored)

    def list_snapshots(self) -> List[dict]:
        return [
            {"index": i, "timestamp": s["timestamp"], "count": s["count"]}
            for i, s in enumerate(self._snapshots)
        ]


# ── Singletons ─────────────────────────────────────────────────────────────────
_audit_instance = None
_snapshot_instance = None

def get_audit_log() -> MemoryAuditLog:
    global _audit_instance
    if _audit_instance is None:
        _audit_instance = MemoryAuditLog()
    return _audit_instance

def get_snapshot() -> MemorySnapshot:
    global _snapshot_instance
    if _snapshot_instance is None:
        _snapshot_instance = MemorySnapshot()
    return _snapshot_instance