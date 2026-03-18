"""
hybrid_retrieval.py — Hybrid Retrieval (ChromaDB Vector + BM25 Keyword)
------------------------------------------------------------------------
எப்படி வேலை செய்யும்:
→ ChromaDB  : Semantic similarity search (meaning-based)
→ BM25      : Keyword exact match search (word-based)
→ RRF Fusion: இரண்டையும் combine பண்ணி best results எடுக்கும்
→ Re-ranking: Final results-ஐ rerank பண்ணும்

Install:
    pip install rank-bm25 chromadb sentence-transformers
"""

import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────
# 1. BM25 Index Builder
# ─────────────────────────────────────────────────────────────

class BM25Index:
    """
    Builds and queries a BM25 index from your existing documents.
    உன் ChromaDB-ல இருக்க same documents-ஐ BM25-க்கும் feed பண்ணு.
    """

    def __init__(self):
        self.bm25 = None
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []

    def build(self, documents: List[str], metadatas: List[Dict] = None):
        """
        documents : list of raw text chunks (same as what you store in ChromaDB)
        metadatas : optional list of metadata dicts
        """
        self.documents = documents
        self.metadatas = metadatas or [{} for _ in documents]

        # Tokenize — simple whitespace split (works well for English + Tamil mixed)
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        logging.info(f"✅ BM25 index built with {len(documents)} documents")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns top_k results with scores."""
        if self.bm25 is None:
            raise ValueError("❌ BM25 index not built yet! Call build() first.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "text": self.documents[idx],
                "metadata": self.metadatas[idx],
                "bm25_score": float(scores[idx]),
                "index": idx
            })

        return results


# ─────────────────────────────────────────────────────────────
# 2. Reciprocal Rank Fusion (RRF)
# ─────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    RRF Formula: score = 1 / (k + rank)
    
    இரண்டு result lists-ஐ combine பண்ணி ஒரே ranked list தரும்.
    k=60 is the standard default (from original RRF paper).
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    # Score from vector search results
    for rank, doc in enumerate(vector_results):
        doc_id = doc["text"][:100]  # Use first 100 chars as unique key
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = doc

    # Score from BM25 results
    for rank, doc in enumerate(bm25_results):
        doc_id = doc["text"][:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sort by combined RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        fused_results.append(doc)

    logging.info(f"🔀 RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(fused_results)} combined")
    return fused_results


# ─────────────────────────────────────────────────────────────
# 3. Main HybridRetriever Class
# ─────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Drop-in replacement for your existing ChromaDB retriever.
    
    Usage:
        retriever = HybridRetriever(chroma_collection, reranker)
        retriever.build_bm25(all_documents, all_metadatas)
        results = retriever.retrieve(query, top_k=5)
    """

    def __init__(self, chroma_collection, reranker=None):
        """
        chroma_collection : உன் existing ChromaDB collection object
        reranker          : உன் existing reranker (optional, but you have one!)
        """
        self.collection = chroma_collection
        self.reranker = reranker
        self.bm25_index = BM25Index()

    def build_bm25(self, documents: List[str], metadatas: List[Dict] = None):
        """Call this once after loading your ChromaDB collection."""
        self.bm25_index.build(documents, metadatas)

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """ChromaDB semantic search — உன் existing logic."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        docs = []
        for i, text in enumerate(results["documents"][0]):
            docs.append({
                "text": text,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "vector_score": 1 - results["distances"][0][i]  # convert distance → similarity
            })
        return docs

    def _rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """உன் existing reranker-ஐ use பண்ணும்."""
        if self.reranker is None:
            return docs[:top_k]

        # Adapt this to match your reranker's exact API
        texts = [doc["text"] for doc in docs]
        reranked = self.reranker.rerank(query, texts)

        reranked_docs = []
        for item in reranked[:top_k]:
            idx = item["index"]
            doc = docs[idx].copy()
            doc["rerank_score"] = item["score"]
            reranked_docs.append(doc)

        return reranked_docs

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Main method — call this instead of your old ChromaDB search.
        
        Flow:
            1. Vector search  (ChromaDB)
            2. Keyword search (BM25)
            3. RRF Fusion     (combine both)
            4. Re-ranking     (உன் existing reranker)
        """
        logging.info(f"🔍 Hybrid retrieval for: '{query[:60]}'")

        # Step 1: Vector Search
        vector_results = self._vector_search(query, top_k=top_k * 2)
        logging.info(f"  📊 Vector results: {len(vector_results)}")

        # Step 2: BM25 Keyword Search
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        logging.info(f"  🔤 BM25 results: {len(bm25_results)}")

        # Step 3: RRF Fusion
        fused = reciprocal_rank_fusion(vector_results, bm25_results)

        # Step 4: Re-ranking (உன் existing reranker)
        final_results = self._rerank(query, fused, top_k=top_k)
        logging.info(f"  ✅ Final results after rerank: {len(final_results)}")

        return final_results


# ─────────────────────────────────────────────────────────────
# 4. How to integrate with your existing code
# ─────────────────────────────────────────────────────────────
"""
BEFORE (உன் existing code):
─────────────────────────────
results = collection.query(query_texts=[query], n_results=5)

AFTER (hybrid):
─────────────────────────────
from hybrid_retrieval import HybridRetriever

# One-time setup (app startup-ல பண்ணு)
retriever = HybridRetriever(
    chroma_collection=collection,   # உன் existing collection
    reranker=your_reranker          # உன் existing reranker
)

# உன் ChromaDB-ல இருக்க எல்லா documents-ஐயும் ஒரு முறை load பண்ணு
all_docs = collection.get()
retriever.build_bm25(
    documents=all_docs["documents"],
    metadatas=all_docs["metadatas"]
)

# Query time — இதை மட்டும் call பண்ணு
results = retriever.retrieve(query, top_k=5)

# results format:
# [
#   {"text": "...", "metadata": {...}, "rrf_score": 0.03, "rerank_score": 0.9},
#   ...
# ]
"""


# ─────────────────────────────────────────────────────────────
# 5. Quick standalone test (ChromaDB இல்லாம test பண்ண)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Fake documents to test BM25 + RRF logic
    sample_docs = [
        "CSE department exam schedule is in May 2025",
        "Fee structure for B.E CSE is 75000 per year",
        "Dr. Ramesh is the HOD of CSE department",
        "Python programming syllabus covers OOP and data structures",
        "College library is open from 9am to 5pm",
        "Internal assessment marks are uploaded in the portal",
        "AI and Machine Learning elective is offered in 7th semester",
        "Campus placement drive starts in November",
    ]

    sample_meta = [{"source": f"doc_{i}"} for i in range(len(sample_docs))]

    # Build BM25 index
    bm25 = BM25Index()
    bm25.build(sample_docs, sample_meta)

    # Test queries
    test_queries = [
        "exam schedule CSE",
        "HOD of computer science",
        "fee payment details",
        "machine learning course",
    ]

    print("\n" + "=" * 60)
    print("🧪 BM25 Search Test")
    print("=" * 60)

    for q in test_queries:
        print(f"\n🔍 Query: {q}")
        results = bm25.search(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['bm25_score']:.3f}] {r['text']}")

    # Test RRF fusion with fake vector results
    print("\n" + "=" * 60)
    print("🔀 RRF Fusion Test")
    print("=" * 60)

    fake_vector = [
        {"text": sample_docs[0], "metadata": {}, "vector_score": 0.92},
        {"text": sample_docs[2], "metadata": {}, "vector_score": 0.85},
        {"text": sample_docs[6], "metadata": {}, "vector_score": 0.78},
    ]
    fake_bm25 = bm25.search("exam CSE department", top_k=3)

    fused = reciprocal_rank_fusion(fake_vector, fake_bm25)
    print(f"\nFused {len(fused)} results:")
    for i, r in enumerate(fused, 1):
        print(f"  {i}. [RRF: {r['rrf_score']:.4f}] {r['text'][:60]}")
