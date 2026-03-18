"""
rag/memory.py — Persistent Conversation Memory (ChromaDB)
----------------------------------------------------------
எப்படி வேலை செய்யும்:
→ ஒவ்வொரு conversation-உம் ChromaDB-ல save ஆகும்
→ New query வந்தா past conversations search பண்ணும்
→ Relevant past context LLM-க்கு pass ஆகும்
→ Server restart ஆனாலும் memory persist ஆகும் ✅
"""

import uuid
import logging
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

MEMORY_PATH = "chroma_db"
MEMORY_COLLECTION = "conversation_memory"

logging.basicConfig(level=logging.INFO)


class PersistentMemory:
    """
    Stores and retrieves conversation history using ChromaDB.
    """

    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.memory_db = Chroma(
            collection_name=MEMORY_COLLECTION,
            persist_directory=MEMORY_PATH,
            embedding_function=self.embeddings
        )
        logging.info("✅ Persistent memory initialized")

    def save(self, question: str, answer: str, session_id: str = "default"):
        """Save a conversation turn to ChromaDB."""
        try:
            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            memory_text = f"User asked: {question}\nAssistant answered: {answer}"

            self.memory_db.add_texts(
                texts=[memory_text],
                metadatas=[{
                    "session_id": session_id,
                    "question": question,
                    "answer": answer[:500],
                    "timestamp": timestamp,
                    "type": "conversation"
                }],
                ids=[doc_id]
            )
            logging.info(f"💾 Memory saved for: '{question[:50]}'")

        except Exception as e:
            logging.error(f"❌ Memory save failed: {e}")

    def recall(self, query: str, session_id: str = "default") -> str:
        """Search past conversations relevant to current query."""
        try:
            results = self.memory_db.similarity_search_with_score(
                query=query,
                k=self.max_results,
                filter={"session_id": session_id}
            )

            if not results:
                return ""

            relevant = [(doc, score) for doc, score in results if score < 1.2]

            if not relevant:
                return ""

            memory_context = "📝 Relevant past conversations:\n"
            for doc, score in relevant:
                memory_context += f"- {doc.page_content}\n"

            logging.info(f"🧠 Memory recalled: {len(relevant)} past conversations")
            return memory_context

        except Exception as e:
            logging.error(f"❌ Memory recall failed: {e}")
            return ""

    def clear(self, session_id: str = "default"):
        """Clear memory for a specific session."""
        try:
            results = self.memory_db.get(where={"session_id": session_id})
            if results["ids"]:
                self.memory_db.delete(ids=results["ids"])
                logging.info(f"🗑️ Memory cleared for session: {session_id}")
        except Exception as e:
            logging.error(f"❌ Memory clear failed: {e}")

    def get_recent(self, session_id: str = "default", limit: int = 5) -> list:
        """Get recent conversations for a session."""
        try:
            results = self.memory_db.get(
                where={"session_id": session_id},
                include=["metadatas"]
            )
            metadatas = results.get("metadatas", [])
            sorted_meta = sorted(metadatas, key=lambda x: x.get("timestamp", ""), reverse=True)
            return sorted_meta[:limit]
        except Exception as e:
            logging.error(f"❌ Get recent failed: {e}")
            return []


# ── Singleton instance ─────────────────────────────────────────────────────────
_memory_instance = None

def get_memory() -> PersistentMemory:
    """Returns a singleton memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = PersistentMemory()
    return _memory_instance


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    memory = get_memory()

    memory.save("What is AI?", "AI is the simulation of human intelligence by machines.", "test_session")
    memory.save("What is deep learning?", "Deep learning is a subset of ML using neural networks.", "test_session")
    memory.save("What are neural networks?", "Neural networks are layers of connected nodes.", "test_session")

    print("\n🧠 Memory Recall Test:\n")
    context = memory.recall("Tell me about machine learning", "test_session")
    print(context)

    print("\n📋 Recent conversations:")
    recent = memory.get_recent("test_session")
    for r in recent:
        print(f"  Q: {r['question'][:60]}")
        print(f"  A: {r['answer'][:60]}")
        print()
        