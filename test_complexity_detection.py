#!/usr/bin/env python3
"""Test complexity detection"""
import requests
import time

print("\n" + "="*70)
print("Testing Complexity Detection")
print("="*70 + "\n")

tests = [
    ("What is Python?", "simple"),
    ("How does machine learning work?", "moderate"),
    ("Compare and contrast supervised with unsupervised learning, explain their benefits and drawbacks, and discuss architectural implications", "complex")
]

for i, (query, expected) in enumerate(tests, 1):
    print(f"\n[Test {i}] {query[:60]}...")
    print(f"Expected: {expected}\n")
    
    try:
        r = requests.post(
            "http://127.0.0.1:8000/agent",
            json={"question": query, "session_id": f"complexity-{i}"},
            timeout=120
        )
        result = r.json()
        refl = result.get("reflection", {})
        
        detected = refl.get("complexity", "N/A")
        good = refl.get("good")
        attempts = result.get("attempts")
        
        print(f"Detected:  {detected}")
        print(f"Good:      {good}")
        print(f"Attempts:  {attempts}")
        print(f"Reason:    {refl.get('reason', 'N/A')}")
        
        status = "✅" if detected == expected else "⚠️"
        print(f"{status} Match: {detected == expected}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    time.sleep(1)

print("\n" + "="*70)
