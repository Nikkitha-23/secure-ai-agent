"""
rag/memory/injector.py — Memory Injection
-------------------------------------------
Hierarchical, token-aware, query-aware injection.
Injects the RIGHT memories in the RIGHT order.
"""

import logging
from typing import List, Tuple
from .types import MemoryEntry, MemoryType
from .scorer import MemoryScorer

logging.basicConfig(level=logging.INFO)

# Max tokens to use for memory injection in prompt
MAX_MEMORY_TOKENS = 300

# Min score to inject a memory
INJECTION_THRESHOLD = {
    MemoryType.PROCEDURAL: 0.5,   # always inject if relevant
    MemoryType.SEMANTIC:   0.55,
    MemoryType.PREFERENCE: 0.45,
    MemoryType.EPISODIC:   0.6,
}

# Injection order — procedural first (most important)
INJECTION_ORDER = [
    MemoryType.PREFERENCE,   # ✅ user context first
    MemoryType.PROCEDURAL,   # ✅ reasoning rules second
    MemoryType.SEMANTIC,     # ✅ facts third
    MemoryType.EPISODIC,     # ✅ past conversations last
]


class MemoryInjector:

    def __init__(self):
        self.scorer = MemoryScorer()

    def _count_tokens(self, text: str) -> int:
        """Approximate token count (1 token ≈ 4 chars)."""
        return len(text) // 4

    def select(
        self,
        query: str,
        entries: List[MemoryEntry],
        max_tokens: int = MAX_MEMORY_TOKENS
    ) -> List[MemoryEntry]:
        """
        Select which memories to inject.
        Hierarchical: Procedural → Semantic → Preference → Episodic
        Token-aware: Stop when token budget is full.
        Query-aware: Only inject if relevance > threshold.
        """
        # Score all entries against current query
        scored = self.scorer.rank(query, entries)

        selected = []
        used_tokens = 0

        # Inject in priority order
        for mtype in INJECTION_ORDER:
            type_entries = [e for e in scored if e.memory_type == mtype]
            threshold = INJECTION_THRESHOLD.get(mtype, 0.5)

            for entry in type_entries:
                # Query-aware: check relevance threshold
                if entry.relevance_score < threshold:
                    continue

                # Token-aware: check budget
                entry_tokens = self._count_tokens(entry.text)
                if used_tokens + entry_tokens > max_tokens:
                    break

                selected.append(entry)
                used_tokens += entry_tokens

        logging.info(f"💉 Injection: selected {len(selected)} memories ({used_tokens} tokens)")
        return selected

    def format(self, entries: List[MemoryEntry]) -> str:
        """
        Format selected memories into prompt-ready string.
        Grouped by type for clarity.
        """
        if not entries:
            return ""

        sections = {}
        for entry in entries:
            t = entry.memory_type.value.capitalize()
            sections.setdefault(t, []).append(entry.text)

        lines = ["📝 Relevant Memory Context:"]
        for mtype in ["Procedural", "Semantic", "Preference", "Episodic"]:
            if mtype in sections:
                lines.append(f"\n[{mtype}]")
                for text in sections[mtype]:
                    lines.append(f"  - {text}")

        return "\n".join(lines)

    def inject(self, query: str, entries: List[MemoryEntry]) -> str:
        """Select + format in one call."""
        selected = self.select(query, entries)
        return self.format(selected)