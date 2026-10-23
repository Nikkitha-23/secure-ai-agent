"""
tests/test_phase3_hybrid_rag.py — Comprehensive Phase 3 Test Suite
-------------------------------------------------------------------
Validates all Phase 3 Hybrid RAG + Evidence Validation criteria:
 1. Dense Retrieval (ChromaDB Vector Search)
 2. BM25 Keyword Search
 3. Reciprocal Rank Fusion (RRF)
 4. Metadata Preservation during Deduplication
 5. CrossEncoder Reranking
 6. Insufficient Evidence Detection
 7. End-to-End /agent RAG Retrieval Pipeline
"""

import sys
import os
import unittest
from langchain_core.documents import Document

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.hybrid_retrieval import (
    BM25Index,
    reciprocal_rank_fusion,
    HybridRetriever,
    get_hybrid_retriever,
    get_doc_hash
)
from rag.evidence import EvidenceValidator, validate_evidence, format_source_attribution
from core.agent import run_agent
from core.executor import Executor


def safe_print(msg: str):
    """Safely print strings avoiding Windows charmap encoding errors."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "ignore").decode("ascii"))


class TestPhase3HybridRAGSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_docs = [
            Document(
                page_content="Deep neural networks use backpropagation to optimize model weights for computer vision tasks.",
                metadata={"source": "deep_learning.pdf", "page": 12, "chunk_id": "dl_01"}
            ),
            Document(
                page_content="Convolutional neural networks (CNN) excel at image classification and feature extraction.",
                metadata={"source": "cnn_guide.pdf", "page": 3, "chunk_id": "cnn_01"}
            ),
            Document(
                page_content="Recurrent neural networks (RNN) and LSTM handle sequential data processing like language translation.",
                metadata={"source": "rnn_guide.pdf", "page": 5, "chunk_id": "rnn_01"}
            ),
            Document(
                page_content="Transformers utilize self-attention mechanisms to outperform RNNs on natural language processing tasks.",
                metadata={"source": "transformers.pdf", "page": 1, "chunk_id": "trans_01"}
            ),
            Document(
                page_content="The college administration office will remain closed during national holidays in December.",
                metadata={"source": "admin_notice.pdf", "page": 2, "chunk_id": "admin_01"}
            )
        ]

    # 1. Dense Retrieval Test
    def test_01_dense_retrieval(self):
        safe_print("\n--- Test 1: Dense Vector Retrieval ---")
        retriever = HybridRetriever(auto_build_bm25=False)

        # Test dense search with synthetic vector results or ChromaDB
        query = "neural networks image classification"
        dense_results = retriever.dense_search(query, top_k=3)
        safe_print(f"Retrieved {len(dense_results)} dense results")

        self.assertIsInstance(dense_results, list)
        if dense_results:
            doc, score = dense_results[0]
            self.assertIsInstance(doc, Document)
            self.assertGreater(score, 0.0)
            safe_print(f"Top dense chunk: {doc.page_content[:60]}... (score={score:.4f})")

    # 2. BM25 Keyword Search Test
    def test_02_bm25_retrieval(self):
        safe_print("\n--- Test 2: BM25 Keyword Retrieval ---")
        bm25_index = BM25Index()
        bm25_index.build(self.sample_docs)

        query = "Convolutional neural networks CNN image"
        results = bm25_index.search(query, top_k=3)

        safe_print(f"BM25 found {len(results)} matches")
        self.assertGreater(len(results), 0)
        top_doc, bm25_score = results[0]
        self.assertIn("Convolutional", top_doc.page_content)
        self.assertGreater(bm25_score, 0.0)
        safe_print(f"Top BM25 result: {top_doc.page_content[:60]}... (score={bm25_score:.4f})")

    # 3. RRF Rank Fusion Test
    def test_03_rrf_merging(self):
        safe_print("\n--- Test 3: Reciprocal Rank Fusion (RRF) ---")
        dense_results = [(self.sample_docs[0], 0.9), (self.sample_docs[1], 0.8)]
        bm25_results = [(self.sample_docs[1], 3.5), (self.sample_docs[3], 2.1)]

        fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)

        safe_print(f"RRF Fused Count: {len(fused)}")
        self.assertEqual(len(fused), 3)  # sample_docs[0], sample_docs[1], sample_docs[3]

        # sample_docs[1] was present in both, so its RRF score should be highest!
        top_fused = fused[0]
        self.assertEqual(top_fused.metadata["chunk_id"], "cnn_01")
        self.assertIn("rrf_score", top_fused.metadata)

        expected_rrf = (1.0 / (60 + 2)) + (1.0 / (60 + 1))  # dense rank 2, bm25 rank 1
        self.assertAlmostEqual(top_fused.metadata["rrf_score"], expected_rrf, places=4)
        safe_print(f"Top RRF fused chunk score: {top_fused.metadata['rrf_score']:.6f}")

    # 4. Metadata Preservation Test
    def test_04_metadata_preservation(self):
        safe_print("\n--- Test 4: Metadata Preservation during Deduplication ---")
        doc_a = Document(
            page_content="Self-attention mechanism in transformers",
            metadata={"source": "paper.pdf", "page": 4, "author": "Vaswani"}
        )
        doc_b = Document(
            page_content="Self-attention mechanism in transformers",
            metadata={"source": "paper.pdf", "page": 4, "chunk_id": "chunk_42", "category": "NLP"}
        )

        fused = reciprocal_rank_fusion([(doc_a, 0.95)], [(doc_b, 4.0)], k=60)

        self.assertEqual(len(fused), 1)
        meta = fused[0].metadata
        safe_print(f"Preserved Metadata: {meta}")

        self.assertEqual(meta["source"], "paper.pdf")
        self.assertEqual(meta["page"], 4)
        self.assertEqual(meta["author"], "Vaswani")
        self.assertEqual(meta["chunk_id"], "chunk_42")
        self.assertIn("rrf_score", meta)
        self.assertIn("vector_score", meta)
        self.assertIn("bm25_score", meta)

    # 5. CrossEncoder Reranking Test
    def test_05_reranking(self):
        safe_print("\n--- Test 5: CrossEncoder Reranking ---")
        executor = Executor()
        query = "What mechanism do Transformers use for natural language processing?"

        reranked = executor.rerank(query, self.sample_docs, top_n=2)
        safe_print(f"Reranked count: {len(reranked)}")

        self.assertEqual(len(reranked), 2)
        top_doc = reranked[0]
        safe_print(f"Top reranked doc: {top_doc.page_content[:70]}...")
        self.assertIn("Transformers", top_doc.page_content)

    # 6. Insufficient Evidence Handling Test
    def test_06_insufficient_evidence(self):
        safe_print("\n--- Test 6: Insufficient Evidence Handling ---")
        validator = EvidenceValidator(min_rerank_threshold=-5.0, min_keyword_match_ratio=0.3)

        query = "What is the orbital velocity of Jupiter's moon Europa?"

        # Test empty docs
        res_empty = validator.validate_evidence(query, [])
        self.assertFalse(res_empty.is_sufficient)
        self.assertEqual(res_empty.confidence_score, 0.0)
        safe_print(f"Empty docs validation: {res_empty.reason}")

        # Test off-topic docs
        off_topic_docs = [self.sample_docs[4]]  # admin_notice.pdf
        res_off_topic = validator.validate_evidence(query, off_topic_docs)
        self.assertFalse(res_off_topic.is_sufficient)
        safe_print(f"Off-topic validation: is_sufficient={res_off_topic.is_sufficient}, reason={res_off_topic.reason}")

    # 7. End-to-End Agent RAG Retrieval Test
    def test_07_end_to_end_agent_retrieval(self):
        safe_print("\n--- Test 7: End-to-End /agent RAG Retrieval ---")
        query = "What are the key topics in machine learning?"

        result = run_agent(query, session_id="test_phase3_e2e")
        safe_print(f"Agent Answer Snippet: {result['answer'][:100]}...")
        safe_print(f"Search Type: {result.get('search_type')}")
        safe_print(f"Sources: {result.get('sources')}")

        self.assertIn("answer", result)
        self.assertTrue(len(result["answer"]) > 0)
        self.assertNotEqual(result.get("search_type"), "security_blocked")


if __name__ == "__main__":
    unittest.main()
