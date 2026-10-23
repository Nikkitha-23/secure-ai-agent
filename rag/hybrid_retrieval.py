"""
rag/hybrid_retrieval.py — Unified Hybrid Retrieval Pipeline
------------------------------------------------------------
Combines:
  1. Dense Vector Search (ChromaDB + HuggingFace Embeddings)
  2. Keyword Exact Match Search (BM25Okapi)
  3. Reciprocal Rank Fusion (RRF) for rank merging
  4. Document Deduplication with Full Metadata Preservation
  5. Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)

Provides a standard LangChain-compatible retriever interface.
"""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RRF_K_DEFAULT = 60


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25."""
    return re.findall(r"\w+", text.lower())


def get_doc_hash(doc: Document) -> str:
    """Generate unique hash identifier for deduplication based on normalized content, source, and page."""
    meta = doc.metadata or {}
    chunk_id = meta.get("chunk_id") or meta.get("id") or ""
    content_clean = doc.page_content.strip()
    source = meta.get("source") or meta.get("file_name") or ""
    page = meta.get("page") or meta.get("page_number") or ""

    if content_clean:
        unique_str = f"{content_clean}::{source}::{page}"
    else:
        unique_str = f"{chunk_id}::{source}::{page}"

    return hashlib.md5(unique_str.encode("utf-8")).hexdigest()


class BM25Index:
    """
    BM25 index manager for keyword retrieval over Document collections.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.docs: List[Document] = []

    def build(self, docs: List[Document]):
        """Build BM25 index from a list of LangChain Document objects."""
        if not docs:
            logger.warning("⚠️ Cannot build BM25 index: empty document list.")
            self.bm25 = None
            self.docs = []
            return

        self.docs = docs
        corpus = [tokenize_text(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"✅ BM25 index built with {len(docs)} documents.")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search BM25 index for top_k documents with BM25 scores."""
        if self.bm25 is None or not self.docs:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score > 0.0:
                doc = self.docs[idx]
                results.append((doc, score))

        return results


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    k: int = RRF_K_DEFAULT
) -> List[Document]:
    """
    Merges dense vector search results and BM25 search results using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum(1 / (k + rank(d)))

    Deduplicates documents while preserving all original metadata.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    dense_scores_map: Dict[str, float] = {}
    bm25_scores_map: Dict[str, float] = {}
    dense_ranks_map: Dict[str, int] = {}
    bm25_ranks_map: Dict[str, int] = {}

    # Process Dense Search Results
    for rank, (doc, vector_score) in enumerate(dense_results):
        doc_id = get_doc_hash(doc)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
        if doc_id not in doc_map:
            doc_map[doc_id] = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
        else:
            for key, value in (doc.metadata or {}).items():
                if key not in doc_map[doc_id].metadata and value is not None:
                    doc_map[doc_id].metadata[key] = value
        dense_scores_map[doc_id] = vector_score
        dense_ranks_map[doc_id] = rank + 1

    # Process BM25 Search Results
    for rank, (doc, bm25_score) in enumerate(bm25_results):
        doc_id = get_doc_hash(doc)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
        bm25_scores_map[doc_id] = bm25_score
        bm25_ranks_map[doc_id] = rank + 1

        if doc_id not in doc_map:
            doc_map[doc_id] = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
        else:
            for key, value in (doc.metadata or {}).items():
                if key not in doc_map[doc_id].metadata and value is not None:
                    doc_map[doc_id].metadata[key] = value

    # Sort documents by RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

    fused_docs: List[Document] = []
    for doc_id in sorted_doc_ids:
        orig_doc = doc_map[doc_id]
        # Create fresh document preserving page_content and updated metadata
        meta_copy = dict(orig_doc.metadata or {})
        meta_copy["rrf_score"] = float(round(rrf_scores[doc_id], 6))
        if doc_id in dense_scores_map:
            meta_copy["vector_score"] = float(round(dense_scores_map[doc_id], 4))
            meta_copy["dense_rank"] = dense_ranks_map[doc_id]
        if doc_id in bm25_scores_map:
            meta_copy["bm25_score"] = float(round(bm25_scores_map[doc_id], 4))
            meta_copy["bm25_rank"] = bm25_ranks_map[doc_id]

        fused_docs.append(Document(page_content=orig_doc.page_content, metadata=meta_copy))

    logger.info(f"🔀 RRF Fusion: merged {len(dense_results)} dense + {len(bm25_results)} BM25 -> {len(fused_docs)} deduplicated documents.")
    return fused_docs


class HybridRetriever:
    """
    Hybrid Retriever combining ChromaDB Vector Search, BM25 Keyword Search,
    RRF Fusion, and CrossEncoder Reranking.
    """

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        auto_build_bm25: bool = True
    ):
        self.chroma_path = chroma_path
        self.embedding_model = embedding_model
        self.vectordb = None
        self.embeddings = None
        self.bm25_index = BM25Index()
        self._initialized = False

        if auto_build_bm25:
            self._initialize()

    def _initialize(self):
        """Initialize embeddings, ChromaDB, and build BM25 index from stored docs."""
        if self._initialized:
            return

        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            self.vectordb = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings
            )
            self.sync_bm25_index()
            self._initialized = True
            logger.info("✅ HybridRetriever initialized successfully.")
        except Exception as e:
            logger.error(f"❌ HybridRetriever initialization error: {e}")
            self._initialized = False

    def sync_bm25_index(self, docs: Optional[List[Document]] = None):
        """Synchronize BM25 index with provided docs or all docs stored in ChromaDB."""
        if docs is not None:
            self.bm25_index.build(docs)
            return

        if not self.vectordb:
            return

        try:
            collection = self.vectordb._collection
            chroma_data = collection.get()

            documents_text = chroma_data.get("documents") or []
            metadatas = chroma_data.get("metadatas") or []
            ids = chroma_data.get("ids") or []

            all_docs = []
            for i, text in enumerate(documents_text):
                meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
                if "chunk_id" not in meta and i < len(ids):
                    meta["chunk_id"] = ids[i]
                all_docs.append(Document(page_content=text, metadata=meta))

            self.bm25_index.build(all_docs)
        except Exception as e:
            logger.error(f"⚠️ Failed to sync BM25 index from ChromaDB: {e}")

    def dense_search(self, query: str, top_k: int = 8) -> List[Tuple[Document, float]]:
        """Perform dense vector search on ChromaDB."""
        if not self.vectordb:
            self._initialize()

        if not self.vectordb:
            return []

        try:
            results_with_score = self.vectordb.similarity_search_with_score(query, k=top_k)
            dense_results = []
            for doc, dist in results_with_score:
                # Convert distance score to similarity score (Chroma default is L2 distance or cosine)
                sim_score = float(1.0 / (1.0 + dist))
                doc_copy = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
                dense_results.append((doc_copy, sim_score))
            return dense_results
        except Exception as e:
            logger.error(f"❌ Dense vector search error: {e}")
            return []

    def bm25_search(self, query: str, top_k: int = 8) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search."""
        return self.bm25_index.search(query, top_k=top_k)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        rerank_top_n: Optional[int] = 3,
        use_reranker: bool = True
    ) -> List[Document]:
        """
        Execute full hybrid retrieval pipeline:
        1. Dense Vector Search
        2. BM25 Keyword Search
        3. RRF Fusion & Deduplication
        4. Cross-Encoder Reranking
        """
        # Step 1 & 2: Dense + BM25
        dense_results = self.dense_search(query, top_k=top_k)
        bm25_results = self.bm25_search(query, top_k=top_k)

        # Step 3: Reciprocal Rank Fusion
        fused_docs = reciprocal_rank_fusion(dense_results, bm25_results, k=RRF_K_DEFAULT)

        if not fused_docs:
            return []

        # Step 4: Reranking
        if use_reranker and rerank_top_n:
            try:
                from rag.reranker import rerank_documents
                reranked_docs = rerank_documents(query, fused_docs, top_n=rerank_top_n)
                return reranked_docs
            except Exception as e:
                logger.error(f"⚠️ Reranker invocation failed in hybrid retrieval: {e}")
                return fused_docs[:rerank_top_n]

        return fused_docs[:top_k]

    def invoke(self, query: str) -> List[Document]:
        """LangChain compatible invoke method."""
        return self.retrieve(query, top_k=8, rerank_top_n=3)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Legacy LangChain retriever interface method."""
        return self.invoke(query)


# Global Singleton Instance for reuse
_global_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get or create singleton HybridRetriever instance."""
    global _global_hybrid_retriever
    if _global_hybrid_retriever is None:
        _global_hybrid_retriever = HybridRetriever()
    return _global_hybrid_retriever
