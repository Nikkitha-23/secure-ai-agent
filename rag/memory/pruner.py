"""
rag/memory/pruner.py — Memory Pruning
---------------------------------------
3 pruning strategies:
    Type-aware TTL   → Each type has different expiry
    Score decay      → Low score memories get removed
    Consolidation    → Similar memories merged into one
"""

import logging
import re
from datetime import datetime
from typing import List, Tuple
from .types import MemoryEntry, MemoryType, MEMORY_TTL, MEMORY_MIN_SCORE

logging.basicConfig(level=logging.INFO)

MAX_MEMORIES_PER_TYPE = {
    MemoryType.EPISODIC:   50,
    MemoryType.SEMANTIC:   30,
    MemoryType.PREFERENCE: 20,
    MemoryType.PROCEDURAL: 20,
}


class MemoryPruner:

    def prune_by_ttl(self, entries: List[MemoryEntry]) -> Tuple[List[MemoryEntry], int]:
        """
        Remove expired memories based on type-aware TTL.
        Procedural memories live longer than episodic ones.
        """
        now = datetime.now().timestamp()
        kept, removed = [], 0

        for entry in entries:
            ttl = MEMORY_TTL.get(entry.memory_type, 86400)
            age = now - entry.created_at
            if age < ttl:
                kept.append(entry)
            else:
                removed += 1
                logging.debug(f"🗑️ TTL pruned [{entry.memory_type.value}]: {entry.text[:40]}")

        if removed:
            logging.info(f"⏰ TTL pruning: removed {removed} expired memories")
        return kept, removed

    def prune_by_score(self, entries: List[MemoryEntry]) -> Tuple[List[MemoryEntry], int]:
        """
        Remove memories with composite score below minimum threshold.
        Each type has its own minimum score.
        """
        kept, removed = [], 0

        for entry in entries:
            min_score = MEMORY_MIN_SCORE.get(entry.memory_type, 0.3)
            if entry.composite_score() >= min_score:
                kept.append(entry)
            else:
                removed += 1
                logging.debug(f"📉 Score pruned [{entry.memory_type.value}]: {entry.text[:40]}")

        if removed:
            logging.info(f"📉 Score pruning: removed {removed} low-score memories")
        return kept, removed

    def prune_by_count(self, entries: List[MemoryEntry]) -> Tuple[List[MemoryEntry], int]:
        """
        Keep only top N memories per type (sorted by composite score).
        """
        by_type = {}
        for entry in entries:
            t = entry.memory_type
            by_type.setdefault(t, []).append(entry)

        kept, removed = [], 0
        for mtype, type_entries in by_type.items():
            max_count = MAX_MEMORIES_PER_TYPE.get(mtype, 30)
            sorted_entries = sorted(type_entries, key=lambda e: e.composite_score(), reverse=True)
            kept.extend(sorted_entries[:max_count])
            removed += max(0, len(sorted_entries) - max_count)

        if removed:
            logging.info(f"🔢 Count pruning: removed {removed} excess memories")
        return kept, removed

    def consolidate(self, entries: List[MemoryEntry]) -> Tuple[List[MemoryEntry], int]:
        """
        Merge very similar memories into one stronger memory.
        Uses simple word overlap to detect duplicates.
        """
        if len(entries) < 2:
            return entries, 0

        consolidated = []
        merged_ids = set()
        merge_count = 0

        for i, entry_a in enumerate(entries):
            if entry_a.id in merged_ids:
                continue

            similar_group = [entry_a]

            for j, entry_b in enumerate(entries):
                if i == j or entry_b.id in merged_ids:
                    continue
                if entry_a.memory_type != entry_b.memory_type:
                    continue
                if self._similarity(entry_a.text, entry_b.text) > 0.75:
                    similar_group.append(entry_b)
                    merged_ids.add(entry_b.id)

            if len(similar_group) > 1:
                # Keep the one with highest composite score
                best = max(similar_group, key=lambda e: e.composite_score())
                # Boost its confidence since it was seen multiple times
                best.confidence_score = min(1.0, best.confidence_score + 0.1 * (len(similar_group) - 1))
                best.access_count += sum(e.access_count for e in similar_group[1:])
                consolidated.append(best)
                merge_count += len(similar_group) - 1
                logging.debug(f"🔗 Consolidated {len(similar_group)} similar memories")
            else:
                consolidated.append(entry_a)

        if merge_count:
            logging.info(f"🔗 Consolidation: merged {merge_count} duplicate memories")
        return consolidated, merge_count

    def _similarity(self, text_a: str, text_b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(re.findall(r'\w+', text_a.lower()))
        words_b = set(re.findall(r'\w+', text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def full_prune(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Run all pruning strategies in order."""
        original = len(entries)

        entries, _ = self.prune_by_ttl(entries)
        entries, _ = self.prune_by_score(entries)
        entries, _ = self.consolidate(entries)
        entries, _ = self.prune_by_count(entries)

        logging.info(f"✂️ Full prune: {original} → {len(entries)} memories")
        return entries