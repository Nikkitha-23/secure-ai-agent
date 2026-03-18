"""
evaluate.py — RAGAS Evaluation for EduBot RAG Pipeline
-------------------------------------------------------
Compatible with ragas==0.1.21 + Groq LLM (no OpenAI needed)

Run:
  1. uvicorn main:app --reload   (Terminal 1)
  2. python evaluate.py          (Terminal 2)
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
import requests, json, os
from dotenv import load_dotenv

load_dotenv()

# ── LLM & Embeddings ───────────────────────────────────────────────────────────
groq_llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

ragas_llm        = LangchainLLMWrapper(groq_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ── Assign LLM to each metric manually ────────────────────────────────────────
faithfulness.llm            = ragas_llm
answer_relevancy.llm        = ragas_llm
context_precision.llm       = ragas_llm
context_recall.llm          = ragas_llm
answer_relevancy.embeddings = ragas_embeddings

# ── Test Questions ─────────────────────────────────────────────────────────────
test_samples = [
    {"question": "What is Artificial Intelligence?",
     "ground_truth": "Artificial Intelligence is the branch of computer science which deals with intelligence of machines where an intelligent agent is a system that takes actions which maximize its chances of success."},
    {"question": "What are the types of Machine Learning algorithms?",
     "ground_truth": "There are three types: Unsupervised Learning, Supervised Learning, and Reinforcement Learning."},
    {"question": "What is Natural Language Processing?",
     "ground_truth": "NLP is the interactions between computers and human language where computers are programmed to process natural languages."},
    {"question": "What are the applications of AI in healthcare?",
     "ground_truth": "Healthcare industries apply AI to make better and faster diagnosis than humans. AI can help doctors with diagnoses and inform when patients are worsening."},
    {"question": "What is a Knowledge Based System?",
     "ground_truth": "A KBS is a computer system capable of giving advice in a particular domain, utilizing knowledge provided by a human expert."},
    {"question": "What is Deep Learning?",
     "ground_truth": "Deep Learning is a subset of machine learning based on artificial neural networks for predictive analysis."},
    {"question": "How is AI used in agriculture?",
     "ground_truth": "Agriculture is applying AI as agriculture robotics, soil and crop monitoring, and predictive analysis."},
    {"question": "What are Neural Networks?",
     "ground_truth": "Neural Networks are biologically inspired systems consisting of a massively connected network of computational neurons organized in layers."},
    {"question": "What is the role of AI in data security?",
     "ground_truth": "AI can be used to make data more safe and secure. Tools like AEG bot and AI2 Platform are used to determine software bugs and cyber-attacks."},
    {"question": "What is the future of AI?",
     "ground_truth": "AI is truly a revolutionary feat of computer science, set to become a core component of all modern software over the coming years and decades."},
    {"question": "What is artificial intelligence according to John McCarthy?",
     "ground_truth": "Artificial intelligence is the science and engineering of making intelligent machines, especially intelligent computer programs."},
    {"question": "What is machine learning in the context of artificial intelligence?",
     "ground_truth": "Machine learning is an application of AI where machines learn from data and experience without being explicitly programmed."},
    {"question": "What is the role of artificial intelligence in smart production?",
     "ground_truth": "Artificial intelligence enables smart production by improving manufacturing processes, optimizing energy resources, enhancing logistics, and improving supply chain management."},
    {"question": "What is Industry 4.0 in relation to artificial intelligence?",
     "ground_truth": "Industry 4.0 refers to the fourth industrial revolution where technologies such as AI, machine learning, IoT, and big data enable smart factories and advanced manufacturing systems."},
    {"question": "How is artificial intelligence used in robotics?",
     "ground_truth": "Artificial intelligence enables robots to perform tasks intelligently, learn from experience, and adapt to different situations without constant human programming."},
    {"question": "What industries commonly apply artificial intelligence technologies?",
     "ground_truth": "Artificial intelligence is widely applied in healthcare, finance, transportation, education, agriculture, and entertainment."},
    {"question": "What is the purpose of artificial intelligence in smart manufacturing?",
     "ground_truth": "Artificial intelligence improves efficiency, sustainability, and productivity in manufacturing by optimizing operations and analyzing large datasets."},
    {"question": "What is the future potential of artificial intelligence?",
     "ground_truth": "Artificial intelligence has the potential to transform industries, enhance decision-making, automate complex tasks, and become a core component of modern software systems."},
    {"question": "How does AI help in fraud detection?",
     "ground_truth": "AI is used in fraud detection by scoring credit applications and monitoring payment card transactions in real time to detect fraudulent activity."},
    {"question": "What is machine vision in artificial intelligence?",
     "ground_truth": "Machine vision allows machines to capture and analyze visual information using cameras, analog to digital conversion, and digital signal processing."},
]

# ── Get answers from FastAPI ───────────────────────────────────────────────────
def get_agent_response(question: str) -> dict:
    try:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question},
            timeout=30
        )
        return response.json()
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return {"answer": "", "sources": []}

# ── Build Dataset ──────────────────────────────────────────────────────────────
def build_eval_dataset() -> Dataset:
    print("\n📊 Collecting answers from EduBot...\n")
    questions, answers, contexts, ground_truths = [], [], [], []

    for sample in test_samples:
        question     = sample["question"]
        ground_truth = sample["ground_truth"]
        print(f"  🔍 Testing: {question[:60]}")

        result  = get_agent_response(question)
        answer  = result.get("answer", "")
        sources = result.get("sources", ["unknown"])

        questions.append(question)
        answers.append(answer)
        context_list = result.get("context", [])
        contexts.append(context_list if context_list else ["no context"])
        ground_truths.append(ground_truth)

        print(f"  ✅ Answer: {answer[:80]}...\n")

    return Dataset.from_dict({
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "ground_truth": ground_truths,
    })

# ── Run Evaluation ─────────────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 60)
    print("🧪 EduBot — RAGAS Evaluation")
    print("=" * 60)

    dataset = build_eval_dataset()
    print("\n⚙️  Running RAGAS metrics...\n")

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n" + "=" * 60)
    print("📊 RAGAS Evaluation Results")
    print("=" * 60)

    metrics_info = {
        "faithfulness"      : ("Faithfulness",      "Hallucination check — answer grounded in context?"),
        "answer_relevancy"  : ("Answer Relevancy",  "Is answer relevant to the question?"),
        "context_precision" : ("Context Precision", "Are retrieved chunks useful?"),
        "context_recall"    : ("Context Recall",    "Were the right chunks retrieved?"),
    }

    overall = []
    for key, (label, desc) in metrics_info.items():
        raw = results[key]
        if isinstance(raw, list):
            valid = [x for x in raw if x is not None and x == x]
            score = round(float(sum(valid) / max(len(valid), 1)), 4) if valid else 0.0
        else:
            score = round(float(raw), 4) if raw == raw else 0.0
        overall.append(score)
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        grade = "🟢 Excellent" if score >= 0.85 else "🟡 Good" if score >= 0.70 else "🔴 Needs Improvement"
        print(f"\n  {label}")
        print(f"  Score : {score:.4f}  {grade}")
        print(f"  [{bar}]")
        print(f"  → {desc}")

    avg = round(sum(overall) / len(overall), 4)
    print(f"\n{'='*60}")
    print(f"  🏆 Overall Score : {avg:.4f}")
    print(f"{'='*60}\n")

    output = {
        "faithfulness"      : overall[0],
        "answer_relevancy"  : overall[1],
        "context_precision" : overall[2],
        "context_recall"    : overall[3],
        "overall"           : avg,
    }
    with open("ragas_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("💾 Results saved to ragas_results.json\n")
    return output

if __name__ == "__main__":
    run_evaluation()
