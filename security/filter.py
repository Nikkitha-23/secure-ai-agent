# security/filter.py

import re

BLOCKED_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"act as",
    r"jailbreak",
    r"developer mode",
    r"you are chatgpt",
    r"bypass safety",
    r"simulate hacking",
    r"reveal hidden",
]

def check_input(user_input: str):
    text = user_input.lower()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text):
            return {
                "status": "block",
                "reason": f"Blocked due to suspicious pattern: {pattern}"
            }

    return {
        "status": "allow",
        "clean_input": user_input
    }