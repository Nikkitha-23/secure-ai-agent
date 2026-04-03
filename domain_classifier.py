"""
domain_classifier.py
---------------------
Hybrid Domain Classifier — Keywords first, LLM fallback.
Classifies any query into: "education" | "healthcare" | "general"

Usage:
    from domain_classifier import classify_domain

    result = classify_domain("What is the attendance policy?")
    print(result)
    # {
    #   "domain": "education",
    #   "confidence": "high",
    #   "method": "keyword",
    #   "matched_keywords": ["attendance", "policy"]
    # }
"""

# NOTE: LLM fallback disabled (no credits). Re-enable by uncommenting llm_classify()
# and adding: import os, json, anthropic

# ─── KEYWORD LISTS ────────────────────────────────────────────────────────────

EDUCATION_KEYWORDS = [
    # Academic
    "syllabus", "semester", "exam", "examination", "schedule", "timetable",
    "assignment", "marks", "grade", "cgpa", "gpa", "result", "revaluation",
    "attendance", "lecture", "lab", "practical", "project", "thesis",
    "dissertation", "viva", "internal", "external", "assessment",
    # Institutional
    "college", "university", "school", "department", "faculty", "dean",
    "professor", "lecturer", "student", "admission", "enrollment", "register",
    "hostel", "campus", "library", "placement", "internship", "scholarship",
    "fee", "tuition", "course", "credit", "unit", "module", "regulation",
    "curriculum", "hall ticket", "backlog", "arrear", "detention",
    # Specific
    "anna university", "vit", "mit", "ffcs", "cat", "fat", "cat1", "cat2",
]

HEALTHCARE_KEYWORDS = [
    # Clinical
    "patient", "doctor", "nurse", "hospital", "clinic", "diagnosis",
    "treatment", "prescription", "medicine", "drug", "dosage", "symptoms",
    "disease", "disorder", "syndrome", "infection", "surgery", "operation",
    "therapy", "medication", "vaccine", "vaccination", "test", "scan",
    "mri", "xray", "x-ray", "ecg", "blood test", "urine test",
    # Medical specialties
    "cardiology", "neurology", "oncology", "pediatrics", "gynecology",
    "orthopedics", "dermatology", "psychiatry", "radiology", "pathology",
    # Health general
    "health", "medical", "clinical", "icu", "emergency", "ward",
    "discharge", "admission", "insurance", "apollo", "aiims", "covid",
    "diabetes", "cancer", "hypertension", "bp", "fever", "pain",
]

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 2   # 2+ keyword matches = high confidence
LOW_CONFIDENCE_THRESHOLD  = 1   # 1 keyword match = low confidence (use LLM to confirm)


# ─── STEP 1: KEYWORD CLASSIFIER ───────────────────────────────────────────────

def keyword_classify(query: str) -> dict:
    """
    Fast keyword-based classification.
    Returns domain + confidence + matched keywords.
    """
    query_lower = query.lower()

    edu_matches = [kw for kw in EDUCATION_KEYWORDS if kw in query_lower]
    hc_matches  = [kw for kw in HEALTHCARE_KEYWORDS if kw in query_lower]

    edu_count = len(edu_matches)
    hc_count  = len(hc_matches)

    # Clear winner
    if edu_count >= HIGH_CONFIDENCE_THRESHOLD and edu_count > hc_count:
        return {
            "domain": "education",
            "confidence": "high",
            "method": "keyword",
            "matched_keywords": edu_matches,
            "edu_score": edu_count,
            "hc_score": hc_count,
        }

    if hc_count >= HIGH_CONFIDENCE_THRESHOLD and hc_count > edu_count:
        return {
            "domain": "healthcare",
            "confidence": "high",
            "method": "keyword",
            "matched_keywords": hc_matches,
            "edu_score": edu_count,
            "hc_score": hc_count,
        }

    # Weak signal — needs LLM
    if edu_count == LOW_CONFIDENCE_THRESHOLD and hc_count == 0:
        return {
            "domain": "education",
            "confidence": "low",
            "method": "keyword",
            "matched_keywords": edu_matches,
            "edu_score": edu_count,
            "hc_score": hc_count,
        }

    if hc_count == LOW_CONFIDENCE_THRESHOLD and edu_count == 0:
        return {
            "domain": "healthcare",
            "confidence": "low",
            "method": "keyword",
            "matched_keywords": hc_matches,
            "edu_score": edu_count,
            "hc_score": hc_count,
        }

    # Ambiguous or no match
    return {
        "domain": "unclear",
        "confidence": "none",
        "method": "keyword",
        "matched_keywords": [],
        "edu_score": edu_count,
        "hc_score": hc_count,
    }


# ─── MAIN: HYBRID CLASSIFIER (LLM disabled — keyword only) ───────────────────

def classify_domain(query: str) -> dict:
    """
    Hybrid classifier — keywords first, LLM fallback for unclear cases.

    Args:
        query: The user's input query string

    Returns:
        dict with keys:
            - domain      : "education" | "healthcare" | "general"
            - confidence  : "high" | "medium" | "low"
            - method      : "keyword" | "llm" | "llm_fallback_failed"
            - matched_keywords : list of matched keywords (if keyword method)
            - reasoning   : explanation (if LLM method)
    """
    if not query or not query.strip():
        return {
            "domain": "general",
            "confidence": "high",
            "method": "empty_query",
            "matched_keywords": [],
        }

    # Step 1: Keyword classification
    keyword_result = keyword_classify(query)

    # High confidence keyword match → return immediately (no API call)
    if keyword_result["confidence"] == "high":
        return keyword_result

    # Step 2: Keyword low-confidence → return as-is with "general" fallback
    if keyword_result["confidence"] == "low":
        return keyword_result

    # No match at all → default to general
    return {
        "domain": "general",
        "confidence": "low",
        "method": "keyword_no_match",
        "matched_keywords": [],
    }


# ─── COLLECTION ROUTER ────────────────────────────────────────────────────────

def get_collection_name(domain: str, tenant_id: str) -> str:
    """
    Map domain + tenant → ChromaDB collection name.

    Args:
        domain    : "education" | "healthcare" | "general"
        tenant_id : "anna_university" | "mit" | "other" | etc.

    Returns:
        ChromaDB collection name string
    """
    if domain == "education":
        return f"education_{tenant_id}"
    elif domain == "healthcare":
        return f"healthcare_{tenant_id}"
    else:
        return f"education_{tenant_id}"  # default to education for now


# ─── TEST ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        # Clear education
        "What is the attendance policy for Anna University?",
        "When is the CS3401 Algorithms exam scheduled?",
        "What is the fee structure for B.Tech CSE?",
        "How many credits does the semester have?",
        # Clear healthcare
        "What is the dosage for paracetamol for fever?",
        "What are the symptoms of diabetes?",
        "How to treat a patient with hypertension?",
        # Ambiguous / general
        "What is machine learning?",
        "Tell me about AI applications",
        "What is Python?",
        # Edge cases
        "",
        "hello",
    ]

    print("=" * 60)
    print("DOMAIN CLASSIFIER — TEST RESULTS")
    print("=" * 60)

    for query in test_queries:
        result = classify_domain(query)
        domain     = result["domain"].upper()
        confidence = result["confidence"]
        method     = result["method"]
        keywords   = result.get("matched_keywords", [])
        reasoning  = result.get("reasoning", "")

        print(f"\nQuery : {query or '(empty)'}")
        print(f"Domain: {domain} | Confidence: {confidence} | Method: {method}")
        if keywords:
            print(f"Keywords: {keywords}")
        if reasoning:
            print(f"Reason: {reasoning}")
        print("-" * 60)