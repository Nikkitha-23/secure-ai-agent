#!/usr/bin/env python3
"""
Test the agent API with 5-word threshold
"""
import requests
import json

print("\n" + "="*50)
print("Testing Agent API (5-word threshold)")
print("="*50 + "\n")

query = "What is machine learning used for?"
session_id = "test-unique-query-001"

payload = {
    "question": query,
    "session_id": session_id
}

print(f"📤 Sending query: {query}\n")

try:
    response = requests.post(
        "http://127.0.0.1:8000/agent",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    
    result = response.json()
    
    print("✅ Response received!\n")
    print(f"Answer: {result.get('answer', 'N/A')}\n")
    print(f"Attempts: {result.get('attempts', 'N/A')}")
    print(f"Reflection: {result.get('reflection', 'N/A')}")
    print(f"Sources: {', '.join(result.get('sources', []))}\n")
    
    # Check if test passed
    reflection = result.get('reflection', {})
    good = reflection.get('good') if reflection else None
    attempts = result.get('attempts')
    
    if attempts == 1 and good == True:
        print("🎯 TEST PASSED! Attempts: 1, Good: true ✓")
    else:
        print(f"⚠️  TEST RESULT: Attempts: {attempts}, Good: {good}")
    
except requests.exceptions.ConnectionError:
    print("❌ Error: Cannot connect to server")
    print("Make sure uvicorn is running on http://127.0.0.1:8000")
except Exception as e:
    print(f"❌ Error: {e}")

print()
