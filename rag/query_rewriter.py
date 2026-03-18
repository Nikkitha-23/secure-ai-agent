"""
query_rewriter.py — Query Rewriting for Better Retrieval
---------------------------------------------------------
User question வந்தா:
→ LLM அதை better search query-ஆ rewrite பண்ணும்
→ Rewritten query ChromaDB / Tavily-க்கு போகும்
→ Better retrieval = Better answer ✅

Example:
  Input : "what about exam?"
  Output: "semester end examination schedule dates and hall ticket details"
"""

from groq import Groq
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)

rewriter_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def rewrite_query(query: str) -> str:
    """
    Takes a user's raw question and rewrites it into
    a better, more detailed search query.

    Args:
        query : original user question

    Returns:
        rewritten query string (or original if rewriting fails)
    """

    prompt = f"""You are a search query optimizer for a college AI assistant.

Your job: Rewrite the user's question into a clear, detailed search query.

Rules:
- Make it more specific and descriptive
- Expand abbreviations (e.g. "exam" → "semester end examination")
- Keep it under 20 words
- Return ONLY the rewritten query — no explanation, no quotes

Original question: {query}
Rewritten query:"""

    try:
        response = rewriter_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50     # just a short query
        )

        rewritten = response.choices[0].message.content.strip()

        # Safety — if rewritten is too long or empty, use original
        if not rewritten or len(rewritten) > 200:
            logging.warning("⚠️ Query rewriting returned bad output → using original")
            return query

        logging.info(f"✏️ Query rewritten: '{query[:40]}' → '{rewritten[:60]}'")
        return rewritten

    except Exception as e:
        logging.error(f"❌ Query rewriting failed: {e} → using original query")
        return query   # safe fallback — original query use பண்ணும்


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "what about exam?",
        "fee?",
        "who is hod?",
        "syllabus for 3rd sem",
        "latest ai stuff",
        "college rules",
    ]

    print("\n✏️ Query Rewriter Test:\n")
    for q in test_queries:
        rewritten = rewrite_query(q)
        print(f"  Original : {q}")
        print(f"  Rewritten: {rewritten}")
        print()
