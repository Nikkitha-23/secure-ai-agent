"""
rag/memory/manager.py — Memory Manager
----------------------------------------
Main class that coordinates all memory operations.
This replaces the old memory.py.

Usage:
    from rag.memory.manager import get_memory_manager
    memory = get_memory_manager()
    memory.save(question, answer, session_id)
    context = memory.recall(query, session_id)
"""

import uuid
import json
import os
import logging
from datetime import datetime
from typing import List, Optional

from .types import MemoryEntry, MemoryType, MEMORY_TTL
from .scorer import MemoryScorer
from .pruner import MemoryPruner
from .validator import MemoryValidator
from .injector import MemoryInjector
from .pipeline import MemoryPipeline
from .observability import get_audit_log, get_snapshot

MEMORY_STORE = "memory_store.json"
logging.basicConfig(level=logging.INFO)


def _classify_memory_type(question: str, answer: str) -> MemoryType:
    """
    Auto-classify memory type based on content.
    """
    text = (question + " " + answer).lower()

    procedural_signals = ["always", "rule", "must", "should", "when you ask", "every time", "pattern"]
    semantic_signals   = ["name is", "i am", "department", "year", "student", "my college", "roll number"]
    preference_signals = ["prefer", "like", "hate", "want", "don't want", "favorite", "better"]

    if any(s in text for s in procedural_signals):
        return MemoryType.PROCEDURAL
    if any(s in text for s in semantic_signals):
        return MemoryType.SEMANTIC
    if any(s in text for s in preference_signals):
        return MemoryType.PREFERENCE
    return MemoryType.EPISODIC


class MemoryManager:

    def __init__(self):
        self._store: List[MemoryEntry] = []
        self.scorer    = MemoryScorer()
        self.pruner    = MemoryPruner()
        self.validator = MemoryValidator()
        self.injector  = MemoryInjector()
        self._load()
        self.pipeline = MemoryPipeline(self._store, self._save)
        logging.info("✅ Advanced Memory Manager initialized")
        self.audit    = get_audit_log()
        self.snapshot = get_snapshot()
        self._write_count = 0
        
    def _load(self):
        if os.path.exists(MEMORY_STORE):
            try:
                with open(MEMORY_STORE, "r") as f:
                    raw = json.load(f)
                self._store = [MemoryEntry.from_dict(d) for d in raw]
                logging.info(f"📂 Loaded {len(self._store)} memories from disk")
            except Exception as e:
                logging.error(f"❌ Memory load failed: {e}")
                self._store = []

    def _save(self):
        try:
            with open(MEMORY_STORE, "w") as f:
                json.dump([e.to_dict() for e in self._store], f, indent=2)
        except Exception as e:
            logging.error(f"❌ Memory save failed: {e}")

    def save(
        self,
        question: str,
        answer: str,
        session_id: str = "default",
        memory_type: Optional[MemoryType] = None
    ):
        """
        Save a conversation to memory.
        Auto-classifies memory type if not provided.
        """
        if not memory_type:
            memory_type = _classify_memory_type(question, answer)

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            text=f"User asked: {question}\nAssistant answered: {answer[:300]}",
            memory_type=memory_type,
            session_id=session_id,
            source_query=question,
            source_answer=answer[:300],
            created_at=datetime.now().timestamp(),
            last_accessed=datetime.now().timestamp(),
        )

        self.scorer.score_all(question, entry)
        self.pipeline.process_write(entry)
        self.audit.log("write", memory_type.value, session_id, question, entry.composite_score())
        # Snapshot every 10 saves
        if self._write_count % 10 == 0:
          self.snapshot.take(self._store)
        logging.info(f"💾 Memory saved [{memory_type.value}]: '{question[:50]}'")

    def recall(self, query: str, session_id: str = "default") -> str:
        """
        Recall and inject relevant memories for a query.
        Returns formatted memory context string.
        """
        # Filter by session
        session_memories = [e for e in self._store if e.session_id == session_id]

        if not session_memories:
            return ""

        # Update access stats for retrieved memories
        for entry in session_memories:
            entry.last_accessed = datetime.now().timestamp()
            entry.access_count += 1

        # Inject relevant memories
        context = self.pipeline.process_read(query, session_id)
        self.audit.log("read", "all", session_id, query)
        return context
    
    def get_procedural_rules(self, query: str, session_id: str = "default") -> list:
        """
        Get procedural memories that should affect agent behavior.
        Used by router to bias retrieval strategy.
        """
        procedural = [
            e for e in self._store
            if e.session_id == session_id
            and e.memory_type == MemoryType.PROCEDURAL
            and e.confidence_score >= 0.5
        ]

        scored = self.scorer.rank(query, procedural)
        return [e.text for e in scored[:3] if e.relevance_score > 0.3]

    def clear(self, session_id: str = "default"):
        """Clear memories for a session."""
        before = len(self._store)
        self._store = [e for e in self._store if e.session_id != session_id]
        removed = before - len(self._store)
        self._save()
        logging.info(f"🗑️ Cleared {removed} memories for session: {session_id}")

    def get_recent(self, session_id: str = "default", limit: int = 5) -> list:
        """Get recent memories for a session."""
        session_memories = [
            e for e in self._store if e.session_id == session_id
        ]
        sorted_memories = sorted(
            session_memories,
            key=lambda e: e.created_at,
            reverse=True
        )
        return [e.to_dict() for e in sorted_memories[:limit]]

    def stats(self, session_id: str = "default") -> dict:
        """Memory statistics."""
        session_memories = [e for e in self._store if e.session_id == session_id]
        by_type = {}
        for e in session_memories:
            t = e.memory_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_memories": len(session_memories),
            "by_type": by_type,
            "avg_confidence": round(
                sum(e.confidence_score for e in session_memories) / max(len(session_memories), 1), 3
            ),
            "avg_importance": round(
                sum(e.importance_score for e in session_memories) / max(len(session_memories), 1), 3
            ),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
_manager_instance = None

def get_memory_manager() -> MemoryManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = MemoryManager()
    return _manager_instance