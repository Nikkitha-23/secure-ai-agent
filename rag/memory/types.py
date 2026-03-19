"""
rag/memory/types.py — Memory Types & Data Structures
------------------------------------------------------
4 Memory Types:
    Episodic   → Past conversations
    Semantic   → Facts about user
    Preference → User preferences
    Procedural → How to behave (rules + patterns)
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class MemoryType(Enum):
    EPISODIC   = "episodic"    # Past conversations
    SEMANTIC   = "semantic"    # Facts about user/domain
    PREFERENCE = "preference"  # User preferences
    PROCEDURAL = "procedural"  # Behavioral rules + patterns


# TTL per memory type (seconds)
MEMORY_TTL = {
    MemoryType.EPISODIC:   86400,    # 1 day
    MemoryType.SEMANTIC:   604800,   # 7 days
    MemoryType.PREFERENCE: 2592000,  # 30 days
    MemoryType.PROCEDURAL: 7776000,  # 90 days
}

# Min score to keep memory alive
MEMORY_MIN_SCORE = {
    MemoryType.EPISODIC:   0.3,
    MemoryType.SEMANTIC:   0.5,
    MemoryType.PREFERENCE: 0.4,
    MemoryType.PROCEDURAL: 0.6,
}


@dataclass
class MemoryEntry:
    """Single memory unit with full scoring metadata."""

    # Core fields
    id: str
    text: str
    memory_type: MemoryType
    session_id: str = "default"

    # Scores (0.0 to 1.0)
    relevance_score: float  = 0.5
    importance_score: float = 0.5
    confidence_score: float = 0.5   # ⭐ NEW — how reliable is this memory?

    # Timestamps
    created_at: float  = field(default_factory=lambda: datetime.now().timestamp())
    last_accessed: float = field(default_factory=lambda: datetime.now().timestamp())
    access_count: int  = 0

    # Metadata
    source_query: str  = ""
    source_answer: str = ""
    tags: list         = field(default_factory=list)

    def composite_score(self, recency_weight: float = 0.3) -> float:
        """
        Combined score for ranking memories.
        Higher = more important to keep/inject.
        """
        now = datetime.now().timestamp()
        age_seconds = now - self.created_at
        recency = max(0.0, 1.0 - (age_seconds / 86400))  # decay over 1 day

        return (
            self.relevance_score  * 0.35 +
            self.importance_score * 0.25 +
            self.confidence_score * 0.25 +
            recency               * recency_weight
        )

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "text":             self.text,
            "memory_type":      self.memory_type.value,
            "session_id":       self.session_id,
            "relevance_score":  self.relevance_score,
            "importance_score": self.importance_score,
            "confidence_score": self.confidence_score,
            "created_at":       self.created_at,
            "last_accessed":    self.last_accessed,
            "access_count":     self.access_count,
            "source_query":     self.source_query,
            "source_answer":    self.source_answer,
            "tags":             self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        d["memory_type"] = MemoryType(d["memory_type"])
        return cls(**d)