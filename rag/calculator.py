"""
rag/calculator.py — Calculator Tool
-------------------------------------
Matches the shape of your existing tools (web_search.py, retrieve.py) so
it plugs into decide_source() / the tool router the same way.

Uses Python's `ast` module to safely evaluate math expressions — NEVER use
eval() directly on user input, even for a "just a calculator" tool.
"""

import ast
import operator
import logging

logging.basicConfig(level=logging.INFO)

# Only these operators are allowed — anything else raises an error.
# This is what makes it safe: there is no path to arbitrary code execution.
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} not allowed")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """
    Safely evaluate a math expression string.
    """
    try:
        expr = expression.strip()
        # Clean conversational prefixes like "Calculate", "What is", etc.
        import re
        expr = re.sub(r"^(calculate|what is|what's|compute|evaluate)\s*", "", expr, flags=re.IGNORECASE).strip().rstrip("?!.")

        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        logging.info(f"🧮 Calculator: {expression} => {expr} = {result}")
        return {
            "answer": str(result),
            "sources": ["calculator"],
            "search_type": "calculator",
            "expression": expr,
            "result": result,
        }
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
        logging.warning(f"🧮 Calculator error on '{expression}': {e}")
        return {
            "answer": f"Could not evaluate '{expression}': {str(e)}",
            "sources": [],
            "search_type": "calculator_error",
            "expression": expression,
            "result": None,
        }


def is_calculator_query(query: str) -> bool:
    """
    Quick heuristic for the Smart Router to detect math expressions.
    Plug this into decide_source() as an early check before pdf/web routing.

    Example wiring in rag/router.py:
        from rag.calculator import is_calculator_query

        def decide_source(query: str) -> str:
            if is_calculator_query(query):
                return "calculator"
            ...existing pdf/web/both logic...
    """
    import re

    # Matches expressions with digits and math operators, e.g. "12 * (3 + 4)"
    math_pattern = re.compile(r"^[\d\s\+\-\*\/\(\)\.\%\^]+$")
    stripped = query.strip().rstrip("?!.")

    if math_pattern.match(stripped) and any(c.isdigit() for c in stripped):
        return True

    # Catches phrasing like "what is 5 + 3" or "calculate 10 * 2"
    trigger_words = ["calculate", "what is", "what's", "compute", "evaluate"]
    lowered = stripped.lower()
    if any(lowered.startswith(w) for w in trigger_words):
        possible_expr = re.sub(r"^(calculate|what is|what's|compute|evaluate)\s*",
                                "", lowered)
        if math_pattern.match(possible_expr.strip()):
            return True

    return False