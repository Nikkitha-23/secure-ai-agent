from rag.retrieve import get_retriever
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Load retriever
retriever = get_retriever()

# Load Groq API
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_question(question):

    # 1️⃣ Retrieve documents
    docs = retriever.invoke(question)

    # 2️⃣ Extract context
    context = "\n\n".join([doc.page_content for doc in docs])

    if len(context.strip()) < 20:
        return {
        "answer": "I could not find relevant information in the provided documents.",
        "sources": []
    }

    # 2️⃣ Extract context
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3️⃣ Create prompt
    prompt = f"""
You are an AI assistant. Use ONLY the provided context to answer.

Context:
{context}

Question:
{question}

Answer clearly:
"""

    # 4️⃣ Call LLM
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = completion.choices[0].message.content

    # 5️⃣ Extract sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    return {
        "answer": answer,
        "sources": list(set(sources))
    }
