"""
rag/memory/validator.py — Memory Validation
---------------------------------------------
3 validators:
    Grounding verifier  → Is memory grounded in facts?
    Deduplication       → Remove exact duplicates
    Conflict resolver   → Handle contradicting memories
"""

import re
import logging
from typing import List, Tuple
from .types import MemoryEntry, MemoryType

logging.basicConfig(level=logging.INFO)

# Words that indicate uncertain/ungrounded memory
UNCERTAINTY_MARKERS = [
    "maybe", "perhaps", "i think", "not sure", "probably",
    "might", "could be", "i guess", "i believe"
]

# Contradiction patterns
CONTRADICTION_PAIRS = [
    ("like", "hate"), ("prefer", "avoid"),
    ("always", "never"), ("yes", "no"),
    ("correct", "incorrect"), ("true", "false")
]


class MemoryValidator:

    def verify_grounding(self, entries: List[MemoryEntry]) -> Tuple[List[MemoryEntry], int]:
        """
        Check if memories are grounded (not speculative).
        Penalize confidence score for uncertain memories.
        """
        flagged = 0
        for entry in entries:
            text_lower = entry.text.lower()
            uncertainty_count = sum(1 for marker in UNCERTAINTY_MARKERS if marker in text_lower)

            if uncertainty_count > 0:
                penalty = min(0.4, uncertainty_count * 0.15)
                entry.confidence_score = max(0.0, entry.confidence_score - penalty)
                flagged += 1
                logging.debug(f"⚠️ Grounding penalty applied: {entry.text[:40]}")

        if flagged:
            logging.info(f"🔍 Grounding check: penalized {flagged} uncertain memories")
        return entries, flagged

    def deduplicate(self, entries: List[MemoryEntry]) -> Tuple[List[MemoryEntry], int]:
        """
        Remove exact or near-exact duplicate memories.
        Keep the one with higher confidence score.
        """
        seen_texts = {}
        deduped = []
        removed = 0

        for entry in entries:
            normalized = re.sub(r'\s+', ' ', entry.text.lower().strip())

            if normalized in seen_texts:
                existing = seen_texts[normalized]
                # Keep higher confidence
                if entry.confidence_score > existing.confidence_score:
                    seen_texts[normalized] = entry
                removed += 1
            else:
                seen_texts[normalized] = entry

        deduped = list(seen_texts.values())

        if removed:
            logging.info(f"♻️ Deduplication: removed {removed} duplicates")
        return deduped, removed

    def resolve_conflicts(self, entries: List[MemoryEntry]) -> Tuple[List[MemoryEntry], int]:
        """
        Detect contradicting memories of same type.
        Keep the one with higher confidence score.
        Mark the loser with reduced confidence.
        """
        conflicts_resolved = 0

        for i, entry_a in enumerate(entries):
            for j, entry_b in enumerate(entries):
                if i >= j:
                    continue
                if entry_a.memory_type != entry_b.memory_type:
                    continue
                if self._are_conflicting(entry_a.text, entry_b.text):
                    conflicts_resolved += 1
                    # Penalize lower confidence memory
                    if entry_a.confidence_score >= entry_b.confidence_score:
                        entry_b.confidence_score = max(0.0, entry_b.confidence_score - 0.3)
                        logging.debug(f"⚡ Conflict: penalized '{entry_b.text[:40]}'")
                    else:
                        entry_a.confidence_score = max(0.0, entry_a.confidence_score - 0.3)
                        logging.debug(f"⚡ Conflict: penalized '{entry_a.text[:40]}'")

        if conflicts_resolved:
            logging.info(f"⚡ Conflict resolution: handled {conflicts_resolved} conflicts")
        return entries, conflicts_resolved

    def _are_conflicting(self, text_a: str, text_b: str) -> bool:
        """Check if two texts contradict each other."""
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()

        for word_a, word_b in CONTRADICTION_PAIRS:
            if word_a in text_a_lower and word_b in text_b_lower:
                return True
            if word_b in text_a_lower and word_a in text_b_lower:
                return True
        return False

    def validate_all(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Run all validation steps."""
        entries, _ = self.deduplicate(entries)
        entries, _ = self.verify_grounding(entries)
        entries, _ = self.resolve_conflicts(entries)
        return entries