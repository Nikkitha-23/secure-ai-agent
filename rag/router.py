"""
router.py — Smart Router (PDF vs Web Search)
--------------------------------------------
Question வந்தா:
→ Math expression? → Calculator use பண்ணும் (LLM call இல்லாம, fast)
→ PDF-ல இருக்கா?  → ChromaDB use பண்ணும்
→ Live info வேணுமா? → Tavily Web Search use பண்ணும்
→ Both வேணுமா?    → இரண்டும் use பண்ணும் ✅
"""

from groq import Groq
from dotenv import load_dotenv
from rag.calculator import is_calculator_query
import logging
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)

router_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def decide_source(query: str) -> str:
    """
    Uses LLM to decide where to search for the answer.

    Returns one of:
        "calculator" → math expression, no LLM/retrieval needed
        "pdf"  → search ChromaDB only
        "web"  → search Tavily only
        "both" → search both and combine
    """

    # 🧮 Catch calculator queries BEFORE calling the LLM — saves an API
    # call + latency for something a regex/heuristic can answer instantly
    if is_calculator_query(query):
        logging.info(f"🧮 Router decision for '{query[:50]}': CALCULATOR")
        return "calculator"

    prompt = f"""You are a routing assistant. Given a user question, decide where to search for the answer.

Rules:
- Reply "pdf"  → if the question is about college documents, syllabus, fees, exams, timetable, rules, faculty, or any internal college information
- Reply "web"  → if the question needs current/live information like today's news, recent events, latest technology, current affairs
- Reply "both" → if the question needs both college info AND current/live information

Reply with ONLY one word: pdf, web, or both. No explanation.

Question: {query}
"""

    try:
        response = router_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5      # we only need one word
        )

        decision = response.choices[0].message.content.strip().lower()

        # Safety check — if unexpected response, default to pdf
        if decision not in ["pdf", "web", "both"]:
            logging.warning(f"⚠️ Unexpected router response: '{decision}' → defaulting to 'pdf'")
            decision = "pdf"

        logging.info(f"🧭 Router decision for '{query[:50]}': {decision.upper()}")
        return decision

    except Exception as e:
        logging.error(f"❌ Router failed: {e} → defaulting to 'pdf'")
        return "pdf"   # safe fallback


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is the exam schedule?",
        "What is the latest news in AI today?",
        "What is our college fee structure and current inflation rate?",
        "Who is the HOD of CSE department?",
        "What are the latest updates in Python 3.13?",
        "What is 12 * 4 + 3?",
        "calculate 100 / 5",
    ]

    print("\n🧭 Router Test:\n")
    for q in test_queries:
        decision = decide_source(q)
        icon = {"pdf": "📄", "web": "🌐", "both": "📄🌐", "calculator": "🧮"}.get(decision, "❓")
        print(f"{icon} [{decision.upper()}] {q}")