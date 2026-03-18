"""
reranker.py — Re-Rank retrieved chunks before sending to LLM
-------------------------------------------------------------
Step 1: Hybrid Retriever gives 10 chunks
Step 2: Re-Ranker scores each chunk against the query
Step 3: Top 3 most relevant chunks go to LLM

Why this helps:
- Removes irrelevant chunks that slipped through retrieval
- LLM gets cleaner context → less hallucination ✅
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)

# ── Load CrossEncoder model (free, runs locally) ───────────────────────────────
# This model scores (query, chunk) pairs → higher score = more relevant
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL)
logging.info(f"✅ Re-Ranker model loaded: {RERANKER_MODEL}")


def rerank_documents(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """
    Takes retrieved docs, scores them against the query,
    returns only the top_n most relevant ones.

    Args:
        query  : user's question
        docs   : list of Document chunks from hybrid retriever
        top_n  : how many chunks to keep (default: 3)

    Returns:
        list of top_n most relevant Document chunks
    """

    if not docs:
        logging.warning("⚠️ No documents to re-rank!")
        return []

    # ── Step 1: Build (query, chunk_text) pairs ────────────────────────────────
    pairs = [(query, doc.page_content) for doc in docs]

    # ── Step 2: Score all pairs ────────────────────────────────────────────────
    scores = reranker.predict(pairs)
    logging.info(f"📊 Re-ranker scores: {[round(s, 3) for s in scores]}")

    # ── Step 3: Sort by score (highest first) ─────────────────────────────────
    scored_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    # ── Step 4: Return only top_n ──────────────────────────────────────────────
    top_docs = [doc for _, doc in scored_docs[:top_n]]

    logging.info(f"✅ Re-ranked: kept top {top_n} out of {len(docs)} chunks")
    return top_docs


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Mock test — replace with real retriever output
    sample_docs = [
        Document(page_content="The exam schedule for November is from 15th to 25th.", metadata={"source": "academic_calendar.pdf"}),
        Document(page_content="The canteen menu includes dosa, idli and rice meals.", metadata={"source": "college_info.pdf"}),
        Document(page_content="End semester exams will be held in the main exam hall.", metadata={"source": "academic_calendar.pdf"}),
        Document(page_content="Students must bring hall tickets to the examination.", metadata={"source": "exam_rules.pdf"}),
        Document(page_content="The library is open from 9 AM to 6 PM on weekdays.", metadata={"source": "college_info.pdf"}),
    ]

    query = "When are the exams scheduled?"
    top_chunks = rerank_documents(query, sample_docs, top_n=3)

    print(f"\n🏆 Top {len(top_chunks)} chunks after re-ranking:\n")
    for i, doc in enumerate(top_chunks):
        print(f"Rank {i+1}: {doc.page_content}")
        print(f"Source : {doc.metadata.get('source','unknown')}")
        print("-" * 50)
