"""
rag/evidence.py — Evidence Validation & Source Attribution Framework
----------------------------------------------------------------------
Validates retrieved documents to determine if they contain sufficient,
relevant evidence to answer the query, preventing hallucinations.
Also formats clean source attributions with preserved metadata.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "upon",
    "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "tell", "me"
}


@dataclass
class EvidenceAssessment:
    is_sufficient: bool
    confidence_score: float
    reason: str
    sources: List[Dict[str, Any]] = field(default_factory=list)


class EvidenceValidator:
    """
    Evaluates evidence sufficiency and source attribution quality.
    """

    def __init__(
        self,
        min_rerank_threshold: float = -7.0,
        min_keyword_match_ratio: float = 0.15
    ):
        self.min_rerank_threshold = min_rerank_threshold
        self.min_keyword_match_ratio = min_keyword_match_ratio

    def format_source_attribution(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Format clean, deduplicated source attributions with preserved metadata.
        """
        seen_sources = set()
        sources = []

        for doc in docs:
            meta = doc.metadata or {}
            source_name = meta.get("source") or meta.get("file_name") or meta.get("title") or "Unknown Document"
            page = meta.get("page") or meta.get("page_number")
            chunk_id = meta.get("chunk_id") or meta.get("id")
            score = meta.get("rerank_score", meta.get("rrf_score", meta.get("vector_score", 0.0)))
            chunk_key = chunk_id or (hashlib.md5(doc.page_content.strip().encode("utf-8")).hexdigest()[:8] if doc.page_content else None)
            key = (source_name, page, chunk_key)
            if key in seen_sources:
                continue
            seen_sources.add(key)

            source_item = {
                "source": source_name,
                "score": float(round(score, 4)) if isinstance(score, (int, float)) else score
            }
            if page is not None:
                source_item["page"] = page
            if chunk_id is not None:
                source_item["chunk_id"] = chunk_id

            sources.append(source_item)

        return sources

    def validate_evidence(
        self,
        query: str,
        docs: List[Document]
    ) -> EvidenceAssessment:
        """
        Validate whether the retrieved documents provide sufficient evidence
        to answer the given query.
        """
        sources = self.format_source_attribution(docs)

        if not docs:
            return EvidenceAssessment(
                is_sufficient=False,
                confidence_score=0.0,
                reason="No documents were retrieved.",
                sources=[]
            )

        # Extract non-stop words from query
        query_words = [
            w.lower().strip("?,!.:;\"'") for w in query.split()
            if w.lower().strip("?,!.:;\"'") and w.lower() not in STOP_WORDS
        ]

        if not query_words:
            # Fallback if all query words were stop words
            query_words = [w.lower() for w in query.split()]

        # Combine text content from docs
        combined_text = " ".join([d.page_content.lower() for d in docs])

        # Check keyword matches
        matched_words = [w for w in query_words if w in combined_text]
        match_ratio = len(matched_words) / max(len(query_words), 1)

        # Reranker score check if available
        top_rerank_score = None
        for doc in docs:
            if "rerank_score" in doc.metadata:
                top_rerank_score = doc.metadata["rerank_score"]
                break

        # Confidence calculation
        if top_rerank_score is not None:
            # Map logit score roughly to 0..1 confidence
            # e.g., rerank_score of 0 -> 0.7, score of 5 -> 0.95, score of -8 -> 0.2
            import math
            sigmoid_score = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, top_rerank_score))))
            confidence = 0.6 * sigmoid_score + 0.4 * match_ratio
        else:
            confidence = match_ratio

        # Determine sufficiency
        if top_rerank_score is not None and top_rerank_score < self.min_rerank_threshold:
            return EvidenceAssessment(
                is_sufficient=False,
                confidence_score=float(round(confidence, 3)),
                reason=f"Top reranker relevance score ({top_rerank_score:.2f}) below threshold ({self.min_rerank_threshold}).",
                sources=sources
            )

        if match_ratio < self.min_keyword_match_ratio and (top_rerank_score is None or top_rerank_score < -2.0):
            return EvidenceAssessment(
                is_sufficient=False,
                confidence_score=float(round(confidence, 3)),
                reason=f"Keyword match ratio ({match_ratio:.2f}) insufficient and low relevance score.",
                sources=sources
            )

        return EvidenceAssessment(
            is_sufficient=True,
            confidence_score=float(round(confidence, 3)),
            reason="Retrieved documents provide sufficient evidence.",
            sources=sources
        )


# Global default validator instance
default_evidence_validator = EvidenceValidator()


def validate_evidence(query: str, docs: List[Document]) -> EvidenceAssessment:
    return default_evidence_validator.validate_evidence(query, docs)


def format_source_attribution(docs: List[Document]) -> List[Dict[str, Any]]:
    return default_evidence_validator.format_source_attribution(docs)
