"""
rag/memory/pipeline.py — Memory Lifecycle Pipeline
----------------------------------------------------
State machine that orchestrates all memory operations.

Flow:
    Input
     ↓
    Memory candidate extraction
     ↓
    Validation layer
     ↓
    Scoring engine
     ↓
    Memory type classifier
     ↓
    Storage decision
     ↓
    Consolidation scheduler
     ↓
    Retrieval orchestrator
     ↓
    Injection planner
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional
from .types import MemoryEntry, MemoryType
from .scorer import MemoryScorer
from .pruner import MemoryPruner
from .validator import MemoryValidator
from .injector import MemoryInjector

logging.basicConfig(level=logging.INFO)

# Consolidation runs every N saves (async)
CONSOLIDATION_INTERVAL = 10


class MemoryPipeline:
    """
    Orchestrates the full memory lifecycle.
    Every memory write/read goes through this pipeline.
    """

    def __init__(self, store_ref: list, save_fn):
        self.store    = store_ref    # reference to manager's _store
        self.save_fn  = save_fn      # manager's _save function
        self.scorer   = MemoryScorer()
        self.pruner   = MemoryPruner()
        self.validator = MemoryValidator()
        self.injector  = MemoryInjector()

        self._write_count = 0
        self._consolidation_lock = threading.Lock()
        logging.info("✅ Memory Pipeline initialized")

    # ── WRITE PIPELINE ─────────────────────────────────────────────────────────

    def process_write(self, entry: MemoryEntry) -> Optional[MemoryEntry]:
        """
        Full write pipeline for a new memory entry.
        Returns processed entry or None if rejected.
        """
        logging.info(f"📥 Pipeline: processing write [{entry.memory_type.value}]")

        # Stage 1: Extract candidate
        entry = self._extract_candidate(entry)
        if not entry:
            return None

        # Stage 2: Validate
        entry = self._validate(entry)
        if not entry:
            logging.info("❌ Pipeline: entry rejected by validator")
            return None

        # Stage 3: Score
        entry = self._score(entry)

        # Stage 4: Storage decision
        if not self._should_store(entry):
            logging.info(f"❌ Pipeline: entry rejected (low score: {entry.composite_score():.2f})")
            return None

        # Stage 5: Add to store
        self.store.append(entry)
        self._write_count += 1
        self.save_fn()

        # Stage 6: Schedule async consolidation
        if self._write_count % CONSOLIDATION_INTERVAL == 0:
            self._schedule_consolidation()

        logging.info(f"✅ Pipeline: memory stored [{entry.memory_type.value}] score={entry.composite_score():.2f}")
        return entry

    def _extract_candidate(self, entry: MemoryEntry) -> Optional[MemoryEntry]:
        """Extract and clean memory candidate."""
        # Skip empty entries
        if not entry.text or len(entry.text.strip()) < 10:
            return None
        # Clean whitespace
        entry.text = " ".join(entry.text.split())
        return entry

    def _validate(self, entry: MemoryEntry) -> Optional[MemoryEntry]:
        """Run validation — grounding + dedup check."""
        # Single entry validation
        temp_list = [entry]
        validated, _ = self.validator.verify_grounding(temp_list)
        validated, _ = self.validator.deduplicate(validated)
        return validated[0] if validated else None

    def _score(self, entry: MemoryEntry) -> MemoryEntry:
        """Score the entry."""
        entry.importance_score = self.scorer.score_importance(entry)
        entry.confidence_score = self.scorer.score_confidence(entry)
        return entry

    def _should_store(self, entry: MemoryEntry) -> bool:
        """Storage decision based on minimum score threshold."""
        from .types import MEMORY_MIN_SCORE
        min_score = MEMORY_MIN_SCORE.get(entry.memory_type, 0.3)
        return entry.composite_score() >= min_score * 0.8  # slight tolerance

    # ── CONSOLIDATION (ASYNC) ──────────────────────────────────────────────────

    def _schedule_consolidation(self):
        """Schedule background consolidation — non-blocking."""
        thread = threading.Thread(
            target=self._run_consolidation,
            daemon=True
        )
        thread.start()
        logging.info("⏰ Consolidation scheduled (async)")

    def _run_consolidation(self):
        """Run full prune in background thread."""
        with self._consolidation_lock:
            logging.info("🔄 Background consolidation starting...")
            before = len(self.store)
            pruned = self.pruner.full_prune(self.store)
            self.store.clear()
            self.store.extend(pruned)
            self.save_fn()
            after = len(self.store)
            logging.info(f"✅ Consolidation complete: {before} → {after} memories")

    # ── READ PIPELINE ──────────────────────────────────────────────────────────

    def process_read(self, query: str, session_id: str) -> str:
        """Read pipeline — retrieve and inject memories into context."""
        logging.info(f"📤 Pipeline: processing read for '{query[:50]}'")

        # Snapshot read — consolidation conflict avoided
        with self._consolidation_lock:
            candidates = [e for e in self.store if e.session_id == session_id]

            if not candidates:
                return ""

            for entry in candidates:
                 entry.last_accessed = datetime.now().timestamp()
                 entry.access_count += 1

            context = self.injector.inject(query, candidates)
            self.save_fn()

        logging.info(f"📤 Pipeline: read complete, context length={len(context)}")
        return context