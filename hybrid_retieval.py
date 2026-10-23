"""
hybrid_retrieval.py (and legacy hybrid_retieval.py) — Root Wrapper
------------------------------------------------------------------
Re-exports the canonical HybridRetriever implementation from rag.hybrid_retrieval.
"""

from rag.hybrid_retrieval import (
    BM25Index,
    reciprocal_rank_fusion,
    HybridRetriever,
    get_hybrid_retriever
)

__all__ = [
    "BM25Index",
    "reciprocal_rank_fusion",
    "HybridRetriever",
    "get_hybrid_retriever"
]
