"""
rag/retrieve.py — Standard Retrieval Module Interface
------------------------------------------------------
Exports get_retriever() returning the canonical HybridRetriever.
"""

from rag.hybrid_retrieval import get_hybrid_retriever, HybridRetriever

CHROMA_PATH = "chroma_db"


def get_retriever() -> HybridRetriever:
    """Returns the canonical HybridRetriever (ChromaDB + BM25 + RRF + Reranker)."""
    return get_hybrid_retriever()


if __name__ == "__main__":
    retriever = get_retriever()
    query = input("Enter your question: ")
    results = retriever.invoke(query)
    print(f"\nTop Hybrid Retrieved Chunks ({len(results)}):\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}:\n")
        print(doc.page_content[:300])
        print(f"Metadata: {doc.metadata}")
        print("\n" + "-" * 50 + "\n")
