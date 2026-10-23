"""
auto_eval_pipeline.py — Auto Evaluation Pipeline
-------------------------------------------------
Every answer-க்கு automatic score போடும்:
✅ Faithfulness   — answer context-ல இருக்கா?
✅ Relevancy      — question-க்கு match ஆகுதா?
✅ Completeness   — போதுமான detail இருக்கா?
✅ Overall Score  — 0.0 to 1.0
"""

from groq import Groq
from dotenv import load_dotenv
import logging
import json
import re
import os

load_dotenv()
logging.basicConfig(level=logging.INFO)

eval_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def score_metric(metric: str, question: str, answer: str, context: str) -> dict:
    prompts = {
        "faithfulness": f"""Score if the answer is faithful to the context.
Context: {context[:500]}
Answer: {answer}
Return ONLY JSON: {{"score": 0.0-1.0, "reason": "one sentence"}}
Score 1.0 = fully from context, 0.0 = hallucinated""",

        "relevancy": f"""Score if the answer is relevant to the question.
Question: {question}
Answer: {answer}
Return ONLY JSON: {{"score": 0.0-1.0, "reason": "one sentence"}}
Score 1.0 = perfectly relevant, 0.0 = completely off-topic""",

        "completeness": f"""Score if the answer is complete and detailed enough.
Question: {question}
Answer: {answer}
Return ONLY JSON: {{"score": 0.0-1.0, "reason": "one sentence"}}
Score 1.0 = fully complete, 0.0 = too vague"""
    }

    try:
        response = eval_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompts[metric]}],
            temperature=0,
            max_tokens=100
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logging.error(f"❌ Scoring failed for {metric}: {e}")

    return {"score": 0.5, "reason": "scoring failed"}

def evaluate(question: str, answer: str, context: str) -> dict:
    logging.info(f"📊 Evaluating: {question[:50]}")

    faithfulness  = score_metric("faithfulness",  question, answer, context)
    relevancy     = score_metric("relevancy",     question, answer, context)
    completeness  = score_metric("completeness",  question, answer, context)

    overall = round(
        (faithfulness["score"] + relevancy["score"] + completeness["score"]) / 3, 2
    )

    if overall >= 0.8:
        grade = "A 🟢"
    elif overall >= 0.6:
        grade = "B 🟡"
    elif overall >= 0.4:
        grade = "C 🟠"
    else:
        grade = "D 🔴"

    result = {
        "overall_score": overall,
        "grade": grade,
        "metrics": {
            "faithfulness":  faithfulness,
            "relevancy":     relevancy,
            "completeness":  completeness
        }
    }

    logging.info(f"✅ Score: {overall} | Grade: {grade}")
    return result