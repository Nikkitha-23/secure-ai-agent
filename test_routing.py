#!/usr/bin/env python3
"""Test improved planner with smart routing"""
import requests

print("\n" + "="*70)
print("Testing Smart Source Routing")
print("="*70 + "\n")

tests = [
    ("What are the latest AI developments in 2026?", "web", "freshness"),
    ("Explain the algorithm for binary search", "pdf", "academic"),
    ("How to install Python and configure a virtual environment", "both", "practical"),
]

for i, (query, expected_primary, query_type) in enumerate(tests, 1):
    print(f"\n[Test {i}] {query_type.upper()}")
    print(f"Query: {query[:65]}...")
    print(f"Expected Primary: {expected_primary}\n")
    
    try:
        r = requests.post(
            "http://127.0.0.1:8000/agent",
            json={"question": query, "session_id": f"routing-{i}"},
            timeout=120
        )
        result = r.json()
        plan = result.get("plan", {})
        route = plan.get("route_info", {})
        
        print(f"DEBUG - Full Plan Keys: {list(plan.keys())}")
        print(f"DEBUG - Route_info: {route}\n")
        
        print(f"✓ Routing Confidence: {route.get('confidence', 'N/A')}")
        print(f"✓ Reasoning: {route.get('reasoning', 'N/A')}")
        print(f"✓ Primary Source: {route.get('primary', 'N/A')}")
        print(f"✓ Suggested Source: {plan.get('source', 'N/A')}")
        
        is_correct = route.get("primary") == expected_primary
        print(f"\n{'✅' if is_correct else '⚠️'} Correct routing: {is_correct}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")

print("\n" + "="*70)
