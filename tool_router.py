"""
FILE: tool_router.py  (or wherever your existing router lives)
--------------------------------------------------------------
Updated Tool Router — registers file_reader alongside existing tools.
Handles tool dispatch and returns result to agent loop.
"""

import json
from tools.file_reader_tool import read_file, TOOL_SCHEMA as FILE_READER_SCHEMA


# ─── 1. TOOL REGISTRY ────────────────────────────────────────────────────────
# Add FILE_READER_SCHEMA here alongside your existing tool schemas.
# Your existing schemas stay exactly as they are.

ALL_TOOL_SCHEMAS = [
    # --- existing tools (keep yours here) ---
    # WEB_SEARCH_SCHEMA,
    # CALCULATOR_SCHEMA,
    # MEMORY_SCHEMA,

    # --- new: file reader ---
    FILE_READER_SCHEMA,
]

# Map tool name → callable function
TOOL_DISPATCH = {
    # "web_search":  web_search,
    # "calculate":   calculate,
    # "recall":      recall_memory,

    "read_file": read_file,   # <-- register here
}


# ─── 2. ROUTE FUNCTION ───────────────────────────────────────────────────────

def route_tool_call(tool_name: str, tool_args: dict) -> str:
    """
    Called by agent_loop.py when the LLM decides to use a tool.
    Returns result as a JSON string (goes back into the LLM context).
    """
    handler = TOOL_DISPATCH.get(tool_name)

    if handler is None:
        return json.dumps({
            "success": False,
            "error": f"Unknown tool: '{tool_name}'. Available: {list(TOOL_DISPATCH.keys())}"
        })

    try:
        result = handler(**tool_args)
        return json.dumps(result, ensure_ascii=False)
    except TypeError as e:
        return json.dumps({
            "success": False,
            "error": f"Bad arguments for tool '{tool_name}': {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Tool '{tool_name}' crashed: {str(e)}"
        })
