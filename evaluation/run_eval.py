"""
run_eval.py — Offline Batch Evaluation Runner
------------------------------------------------
Sample questions-ஐ /ask endpoint-க்கு run பண்ணி,
ஒவ்வொன்னுக்கும் faithfulness/relevancy/completeness score போடும்.
Results → metrics/eval_report.json
"""

import requests
import json
import os
import logging
from datetime import datetime
from auto_eval_pipeline import evaluate

logging.basicConfig(level=logging.INFO)

API_URL = "http://127.0.0.1:8000/ask"
METRICS_DIR = "metrics"
EVAL_REPORT = os.path.join(METRICS_DIR, "eval_report.json")
os.makedirs(METRICS_DIR, exist_ok=True)

# Sample test questions — mix of PDF-grounded, web-search, and tricky/hallucination-prone
TEST_QUESTIONS = [
    "What is the attendance policy?",
    "What is RAG in AI?",
    "Compare RAG and fine-tuning approaches",
    "What is the fee structure for B.Tech?",
    "Explain transformers in deep learning",
    "What is the latest news in AI today?",
    "Calculate the average of 85, 90, 78",
    "What is machine learning?",
]


def run_batch_eval(session_id: str = "eval_runner"):
    results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        logging.info(f"\n[{i}/{len(TEST_QUESTIONS)}] Testing: {question}")

        try:
            response = requests.post(
                API_URL,
                json={"question": question, "session_id": session_id},
                timeout=60
            )
            data = response.json()

            answer = data.get("answer", "")
            context_list = data.get("context", [])
            context = "\n\n".join(context_list) if context_list else ""
            grounding = data.get("grounding", {})

            # Run LLM-based eval (faithfulness, relevancy, completeness)
            eval_result = evaluate(question, answer, context)

            result_entry = {
                "question": question,
                "answer": answer[:300],
                "search_type": data.get("search_type", "unknown"),
                "grounding_label": grounding.get("label", "n/a"),
                "grounding_score": grounding.get("score", None),
                "eval_overall_score": eval_result["overall_score"],
                "eval_grade": eval_result["grade"],
                "eval_metrics": eval_result["metrics"],
                "timestamp": datetime.now().isoformat()
            }
            results.append(result_entry)

            logging.info(f"  Grounding: {grounding.get('label', 'n/a')} | Eval Grade: {eval_result['grade']}")

        except Exception as e:
            logging.error(f"  ❌ Failed: {e}")
            results.append({
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    # Compute summary stats
    valid_results = [r for r in results if "error" not in r]
    summary = {
        "total_tested": len(TEST_QUESTIONS),
        "successful": len(valid_results),
        "avg_eval_score": round(
            sum(r["eval_overall_score"] for r in valid_results) / max(len(valid_results), 1), 3
        ) if valid_results else 0,
        "grounding_distribution": {},
        "grade_distribution": {},
        "last_run": datetime.now().isoformat()
    }

    for r in valid_results:
        g_label = r.get("grounding_label", "n/a")
        summary["grounding_distribution"][g_label] = summary["grounding_distribution"].get(g_label, 0) + 1
        grade = r.get("eval_grade", "n/a")
        summary["grade_distribution"][grade] = summary["grade_distribution"].get(grade, 0) + 1

    report = {
        "summary": summary,
        "results": results
    }

    with open(EVAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info(f"\n{'='*50}")
    logging.info(f"✅ Eval complete! Avg score: {summary['avg_eval_score']} | Report saved to {EVAL_REPORT}")
    logging.info(f"{'='*50}")

    return report


if __name__ == "__main__":
    run_batch_eval()