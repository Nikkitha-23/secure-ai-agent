"""
web_search.py — Real-time Web Search using Tavily
--------------------------------------------------
PDF-ல இல்லாத questions-க்கு
live internet-ல இருந்து answer எடுக்கும் ✅
"""

from tavily import TavilyClient
from langchain_core.documents import Document
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)

# ── Tavily Client ──────────────────────────────────────────────────────────────



def web_search(query: str, max_results: int = 3) -> list[Document]:
    """
    Searches the web using Tavily and returns results
    as LangChain Document objects (same format as ChromaDB)

    Args:
        query       : user's question
        max_results : how many web results to fetch (default: 3)

    Returns:
        list of Document chunks from web
    """
    try:
        logging.info(f"🌐 Web search triggered for: {query}")

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        logging.info(f"🌐 Web search triggered for: {query}")
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,        # Tavily gives a direct answer too
            search_depth="advanced"     # deeper search = better results
        )

        docs = []

        # ── Direct answer from Tavily (most useful) ────────────────────────────
        if response.get("answer"):
            docs.append(Document(
                page_content=response["answer"],
                metadata={"source": "Tavily Web Search (Direct Answer)", "type": "web"}
            ))

        # ── Individual search results ──────────────────────────────────────────
        for result in response.get("results", []):
            docs.append(Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": result.get("url", "Web"),
                    "title":  result.get("title", ""),
                    "type":   "web"
                }
            ))

        logging.info(f"✅ Web search returned {len(docs)} results")
        return docs

    except Exception as e:
        logging.error(f"❌ Web search failed: {e}")
        return []


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = input("Enter search query: ")
    results = web_search(query)

    print(f"\n🌐 Web Search Results ({len(results)}):\n")
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.metadata.get('source')}")
        print(doc.page_content[:300])
        print("-" * 50)
