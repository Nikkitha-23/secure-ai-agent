"""
rag/memory/scorer.py — Memory Scoring
---------------------------------------
4 scores:
    Relevance   → How related to current query?
    Recency     → How recent?
    Importance  → How useful overall?
    Confidence  → How reliable/accurate?
"""

import re
import logging
from datetime import datetime
from typing import List
from .types import MemoryEntry, MemoryType

logging.basicConfig(level=logging.INFO)


class MemoryScorer:

    # Keywords that signal high importance per type
    IMPORTANCE_KEYWORDS = {
        MemoryType.EPISODIC:   ["exam", "fee", "deadline", "urgent", "important"],
        MemoryType.SEMANTIC:   ["name", "department", "year", "roll", "student"],
        MemoryType.PREFERENCE: ["always", "never", "prefer", "like", "hate", "want"],
        MemoryType.PROCEDURAL: ["rule", "always", "must", "should", "when", "if"],
    }

    def score_relevance(self, query: str, entry: MemoryEntry) -> float:
        """
        How relevant is this memory to the current query?
        Simple keyword overlap — fast and effective.
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        memory_words = set(re.findall(r'\w+', entry.text.lower()))

        if not query_words or not memory_words:
            return 0.0

        overlap = query_words & memory_words
        score = len(overlap) / max(len(query_words), 1)
        return min(1.0, score * 2)   # boost small overlaps

    def score_recency(self, entry: MemoryEntry) -> float:
        """
        How recent is this memory?
        Decays over the TTL period of its type.
        """
        from .types import MEMORY_TTL
        now = datetime.now().timestamp()
        age = now - entry.created_at
        ttl = MEMORY_TTL.get(entry.memory_type, 86400)
        return max(0.0, 1.0 - (age / ttl))

    def score_importance(self, entry: MemoryEntry) -> float:
        """
        How important is this memory overall?
        Based on keywords + access count.
        """
        keywords = self.IMPORTANCE_KEYWORDS.get(entry.memory_type, [])
        text_lower = entry.text.lower()

        keyword_hits = sum(1 for kw in keywords if kw in text_lower)
        keyword_score = min(1.0, keyword_hits / max(len(keywords), 1))

        # Boost frequently accessed memories
        access_boost = min(0.3, entry.access_count * 0.05)

        return min(1.0, keyword_score * 0.7 + access_boost + 0.2)

    def score_confidence(self, entry: MemoryEntry) -> float:
        """
        How confident are we this memory is accurate?
        Based on:
        - Source answer length (longer = more detailed = more confident)
        - Access count (verified multiple times = more confident)
        - Memory type (procedural = high confidence needed)
        """
        # Longer answers = more detailed = higher confidence
        answer_len = len(entry.source_answer.split()) if entry.source_answer else 0
        length_score = min(1.0, answer_len / 50)

        # More accesses = more validated
        access_score = min(0.3, entry.access_count * 0.1)

        # Procedural memories need high confidence
        type_bonus = 0.2 if entry.memory_type == MemoryType.PROCEDURAL else 0.0

        return min(1.0, length_score * 0.5 + access_score + type_bonus + 0.2)

    def score_all(self, query: str, entry: MemoryEntry) -> MemoryEntry:
        """Update all scores for an entry given current query."""
        entry.relevance_score  = self.score_relevance(query, entry)
        entry.importance_score = self.score_importance(entry)
        entry.confidence_score = self.score_confidence(entry)
        return entry

    def rank(self, query: str, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Rank memories by composite score for a given query."""
        scored = []
        for entry in entries:
            self.score_all(query, entry)
            scored.append((entry.composite_score(), entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored]