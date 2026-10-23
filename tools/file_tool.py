"""
tools/file_tool.py — File Reader Tool
--------------------------------------
Agent-callable 5th tool:
  - list   → uploaded PDFs பாரு
  - read   → full text extract பண்ணு
  - summarize → LLM summary தரும்
  - ask    → PDF பத்தி specific question கேக்கலாம்
"""

import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = "data/uploads"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Keywords for agent intent detection ───────────────────────────────────────
FILE_KEYWORDS = [
    "summarize pdf",
    "summarize file",
    "summarize the pdf",
    "summarize the file",
    "summarize uploaded",
    "read file",
    "read pdf",
    "read the pdf",
    "read the file",
    "uploaded pdf",
    "uploaded file",
    "what does this pdf say",
    "what does the pdf say",
    "what is in the pdf",
    "list files",
    "list uploaded",
    "show uploaded",
    "which files",
    "available pdfs",
]


def is_file_intent(query: str) -> bool:
    """Returns True if query is asking about an uploaded file."""
    q = query.lower().strip()
    if any(kw in q for kw in FILE_KEYWORDS):
        return True
    file_words = ["pdf", "file", "document", "uploaded"]
    action_words = ["list", "show", "summarize", "read", "what", "which", "available"]
    return any(w in q for w in file_words) and any(w in q for w in action_words)


# ── Core Functions ─────────────────────────────────────────────────────────────

def get_uploaded_files() -> list:
    """Returns list of uploaded PDF filenames in data/uploads/."""
    if not os.path.exists(UPLOAD_DIR):
        return []
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]


def read_pdf(filename: str) -> str:
    """Extracts full text from a saved PDF file."""
    from langchain_community.document_loaders import PyPDFLoader

    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return f"❌ File not found: {filename}. Use 'list' action to see available files."

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        logging.info(f"📄 read_pdf: {filename} → {len(full_text)} chars extracted")
        return full_text
    except Exception as e:
        logging.error(f"❌ read_pdf failed for {filename}: {e}")
        return f"❌ Could not read {filename}: {str(e)}"


def summarize_pdf(filename: str) -> str:
    """Summarizes a PDF using LLM."""
    full_text = read_pdf(filename)
    if full_text.startswith("❌"):
        return full_text

    # Truncate to avoid token overflow (keep first 6000 chars)
    truncated = full_text[:6000]
    if len(full_text) > 6000:
        logging.info(f"⚠️ summarize_pdf: {filename} truncated to 6000 chars (original: {len(full_text)})")

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Summarize the given document clearly "
                        "and concisely. ALWAYS respond in English. Cover the main topics, "
                        "key points, and important details."
                    )
                },
                {
                    "role": "user",
                    "content": f"Summarize this document:\n\n{truncated}"
                }
            ],
            temperature=0.3,
            max_tokens=600
        )
        summary = response.choices[0].message.content.strip()
        logging.info(f"✅ summarize_pdf: {filename} → summary generated")
        return summary
    except Exception as e:
        logging.error(f"❌ summarize_pdf failed: {e}")
        return f"❌ Summarization failed: {str(e)}"


def ask_pdf(filename: str, question: str) -> str:
    """Answers a specific question about a PDF using its full text."""
    full_text = read_pdf(filename)
    if full_text.startswith("❌"):
        return full_text

    truncated = full_text[:6000]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer questions based ONLY on the "
                        "provided document. If the answer is not found in the document, "
                        "clearly say so. ALWAYS respond in English."
                    )
                },
                {
                    "role": "user",
                    "content": f"Document:\n{truncated}\n\nQuestion: {question}"
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        logging.info(f"✅ ask_pdf: {filename} | Q: {question[:40]}...")
        return answer
    except Exception as e:
        logging.error(f"❌ ask_pdf failed: {e}")
        return f"❌ Ask failed: {str(e)}"


# ── Tool Class (Agent Interface) ───────────────────────────────────────────────

class FileTool:
    """
    Agent-callable File Tool.

    Usage:
        file_tool.run(action="list")
        file_tool.run(action="summarize", filename="notes.pdf")
        file_tool.run(action="read",      filename="notes.pdf")
        file_tool.run(action="ask",       filename="notes.pdf", question="What is chapter 2 about?")
    """

    def run(self, action: str, filename: str = None, question: str = None) -> str:

        # ── LIST ──────────────────────────────────────────────────────────────
        if action == "list":
            files = get_uploaded_files()
            if not files:
                return "No PDFs uploaded yet. Use /upload to add files."
            return "📂 Uploaded PDFs:\n" + "\n".join(f"  • {f}" for f in files)

        # ── READ ──────────────────────────────────────────────────────────────
        elif action == "read":
            if not filename:
                return "❌ filename required for read action."
            return read_pdf(filename)

        # ── SUMMARIZE ─────────────────────────────────────────────────────────
        elif action == "summarize":
            if not filename:
                # Auto-pick if only one PDF uploaded
                files = get_uploaded_files()
                if len(files) == 0:
                    return "❌ No PDFs uploaded yet."
                elif len(files) == 1:
                    filename = files[0]
                    logging.info(f"📄 Auto-selected: {filename}")
                else:
                    file_list = "\n".join(f"  • {f}" for f in files)
                    return (
                        f"❌ Multiple PDFs found. Specify which one:\n{file_list}\n\n"
                        "Example: 'Summarize student_handbook.pdf'"
                    )
            return summarize_pdf(filename)

        # ── ASK ───────────────────────────────────────────────────────────────
        elif action == "ask":
            if not question:
                return "❌ question required for ask action."
            if not filename:
                # Auto-pick if only one PDF uploaded
                files = get_uploaded_files()
                if len(files) == 0:
                    return "❌ No PDFs uploaded yet."
                elif len(files) == 1:
                    filename = files[0]
                    logging.info(f"📄 Auto-selected: {filename}")
                else:
                    file_list = "\n".join(f"  • {f}" for f in files)
                    return (
                        f"❌ Multiple PDFs found. Specify which one:\n{file_list}"
                    )
            return ask_pdf(filename, question)

        else:
            return f"❌ Unknown action: '{action}'. Valid: list, read, summarize, ask"


# ── Singleton for agent import ─────────────────────────────────────────────────
file_tool = FileTool()


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tool = FileTool()

    print("=== LIST ===")
    print(tool.run(action="list"))

    print("\n=== INTENT DETECTION ===")
    test_queries = [
        "Summarize the uploaded PDF",
        "What is the exam schedule?",
        "Read the file please",
        "What is AI?",
        "List uploaded files",
    ]
    for q in test_queries:
        intent = "📄 FILE" if is_file_intent(q) else "🔀 ROUTER"
        print(f"{intent} → {q}")
