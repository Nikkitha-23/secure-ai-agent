# security/filter.py
import re

BLOCKED_PATTERNS = [
    # Prompt injection
    r"ignore previous instructions",
    r"ignore all (previous |prior )?instructions",
    r"ignore (the |all |your )?rules",
    r"disregard (previous |prior |all )?instructions",
    r"forget (everything|all|previous|your instructions)",
    r"override (your |all )?instructions",
    r"bypass (safety|filter|security|rules)",

    # Jailbreak attempts
    r"jailbreak",
    r"developer mode",
    r"dan mode",
    r"act as (if you are|a|an)",
    r"pretend (you are|to be)",
    r"you are (now |)?(chatgpt|gpt|an? ai without|unrestricted)",
    r"simulate (hacking|attack|exploit)",

    # System prompt extraction
    r"system prompt",
    r"reveal (your |hidden |)?(prompt|instructions|rules)",
    r"what (are your|is your) (instructions|prompt|rules)",
    r"show me your (prompt|instructions|system)",

    # Role manipulation
    r"you (are|will be|must be) (now |)?(a|an) (hacker|criminal|unrestricted)",
    r"switch (to|into) (a |an )?(different|unrestricted|evil) mode",

    # Instruction override
    r"new instructions?:",
    r"updated instructions?:",
    r"from now on (you|ignore|forget)",
    r"start over (and|with|from)",
]

# Suspicious but not auto-block — reduce confidence
SUSPICIOUS_PATTERNS = [
    r"say hello",
    r"repeat after me",
    r"tell me a secret",
    r"what (can|can't) you do",
]


def check_input(user_input: str):
    text = user_input.lower().strip()

    # Block check
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text):
            return {
                "status": "block",
                "reason": f"Blocked: suspicious pattern detected [{pattern}]",
                "clean_input": ""
            }

    # Suspicious check — allow but flag
    suspicious = []
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text):
            suspicious.append(pattern)

    return {
        "status": "allow",
        "clean_input": user_input,
        "suspicious": suspicious  # log but don't block
    }