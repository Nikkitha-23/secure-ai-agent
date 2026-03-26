#!/usr/bin/env python3
"""
Test complexity detection in reflect() function
"""
import requests
import json

print("\n" + "="*60)
print("Testing Complexity Detection in Agent")
print("="*60 + "\n")

# Test cases: simple, moderate, complex
test_queries = [
    {
        "query": "What is Python?",
        "expected_complexity": "simple",
        "description": "Simple one-word query"
    },
    {
        "query": "How does machine learning work in neural networks?",
        "expected_complexity": "moderate",
        "description": "Moderate multi-part question"
    },
    {
        "query": "Compare and contrast supervised vs unsupervised learning algorithms, including their advantages, disadvantages, and real-world applications in healthcare, finance, and technology.",
        "expected_complexity": "complex",
        "description": "Complex multi-faceted question with analysis"
    }
]

for idx, test_case in enumerate(test_queries, 1):
    print(f"📋 Test {idx}: {test_case['description']}")
    print(f"Query: {test_case['query']}\n")
    
    payload = {
        "question": test_case["query"],
        "session_id": f"complexity-test-{idx}"
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/agent",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        reflection = result.get('reflection', {})
        
        print(f"✅ Response received!")
        print(f"Complexity: {reflection.get('complexity', 'N/A')}")
        print(f"Attempts: {result.get('attempts')}")
        print(f"Good: {reflection.get('good')}")
        print(f"Reason: {reflection.get('reason', 'N/A')}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}\n")

print("="*60)
print("✅ Complexity detection testing complete!")
print("="*60 + "\n")
