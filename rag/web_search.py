from ddgs import DDGS
from langchain_core.documents import Document
from dotenv import load_dotenv
import logging
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)

def web_search(query: str, max_results: int = 3) -> list[Document]:
    try:
        logging.info(f"🌐 Web search triggered for: {query}")
        
        # ✅ Tavily first try
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            response = client.search(query=query, max_results=max_results, include_answer=True, search_depth="advanced")
            docs = []
            if response.get("answer"):
                docs.append(Document(page_content=response["answer"], metadata={"source": "Tavily Direct Answer", "type": "web"}))
            for result in response.get("results", []):
                docs.append(Document(page_content=result.get("content", ""), metadata={"source": result.get("url", "Web"), "title": result.get("title", ""), "type": "web"}))
            logging.info(f"✅ Tavily returned {len(docs)} results")
            return docs
        
        # ⚡ Tavily fail ஆனா DuckDuckGo fallback
        except Exception as tavily_err:
            logging.warning(f"⚠️ Tavily failed: {tavily_err} → switching to DuckDuckGo")
            docs = []
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            for result in results:
                docs.append(Document(page_content=result.get("body", ""), metadata={"source": result.get("href", "Web"), "title": result.get("title", ""), "type": "web"}))
            logging.info(f"✅ DuckDuckGo returned {len(docs)} results")
            return docs

    except Exception as e:
        logging.error(f"❌ Web search failed: {e}")
        return []