"""
core/planner.py — Unified Planning Engine
-------------------------------------------
Merges the best from both agent_loop.py files into a single module:
  - Intent detection (tool routing)
  - Complexity detection
  - Source routing (pdf/web/both)
  - Task decomposition (multi-step planning)

All planning decisions happen HERE, not scattered across files.

Usage:
    from core.planner import Planner
    planner = Planner()
    plan = planner.plan(query)
"""

import re
import json
import logging
from typing import Optional

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from app.config import settings, call_llm

logger = logging.getLogger(__name__)


@dataclass
class AgentPlan:
    original_query: str
    intent: str
    complexity_level: str
    search_type: str
    recommended_tool: str
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    needs_rewrite: bool = False
    max_retries: int = 1
    raw_plan_dict: Dict[str, Any] = field(default_factory=dict)



# ── Intent Detection ─────────────────────────────────────────────────────────

class IntentType:
    """Known intent categories."""
    GREETING = "greeting"
    CALCULATOR = "calculator"
    FILE = "file"
    IMAGE = "image"
    SCIENCE = "science"
    PYTHON = "python"
    WEB = "web"
    PDF = "pdf"
    BOTH = "both"
    TRANSFORMATION = "transformation"
    GENERAL = "general"



GREETING_EXACT_PATTERNS = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "greetings", "hi there", "hello there", "hey there", "howdy",
    "how are you", "how are you doing", "how is it going", "hows it going",
    "how is your day", "how was your day", "how do you do",
    "what's up", "whats up", "what is up", "whatsapp", "what are you doing", "what are you up to",
    "have you eaten", "have you eaten yet", "did you eat", "did you eat yet",
    "have you had lunch", "have you had dinner", "did you have lunch", "did you have dinner",
    "nice to meet you", "good to see you", "thank you", "thanks", "thanks a lot", "thank you so much",
    "bye", "goodbye", "see you", "see ya", "have a good day", "have a nice day",
    "who are you", "what is your name", "whats your name", "who created you", "who made you",
    "what can you do", "what do you do", "help",
    "are you human", "are you an ai", "are you a bot", "are you real", "are you hungry", "are you busy",
    "tell me a joke", "say something funny", "how do you feel"
}

GREETING_PREFIXES = (
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "greetings", "hi there", "hello there", "hey there", "howdy"
)

CASUAL_REGEX_PATTERNS = [
    r"^have you (eaten|had|slept)",
    r"^did you (eat|have|sleep)",
    r"^(how|hows) (are you|is it going|was your day|is your day|are things|do you feel)",
    r"^(what|whats) (is your name|are you doing|are you up to|can you do)",
    r"^(who) (are you|created you|made you)",
    r"^(are you) (human|an ai|a bot|real|hungry|busy|alive|single)",
    r"^(tell me|say) (a joke|something funny)",
    r"^(thank you|thanks|bye|goodbye|see you)",
]

DOCUMENT_EXPLICIT_KEYWORDS = {
    "pdf", "file", "document", "documents", "uploaded", "paper", "report", "chapter",
    "section", "textbook", "notes", "policy", "tuition", "admission", "requirements",
    "database", "record", "article", "manual", "guide", "syllabus", "course"
}

RAG_INTENT_KEYWORDS = DOCUMENT_EXPLICIT_KEYWORDS.union({
    "summarize", "compare", "contrast", "analysis", "details"
})


def is_greeting_query(query: str) -> bool:
    """
    Check if a query is a pure greeting or casual conversational message.
    Fast heuristic — no LLM call. Returns True if query is a greeting / casual small talk.
    If query contains explicit document keywords or document requests, returns False.
    """
    cleaned = re.sub(r"[^\w\s]", "", query.lower()).strip()
    if not cleaned:
        return False

    # 1. If query contains explicit document/RAG keywords, it is NOT a simple greeting
    if any(kw in cleaned for kw in DOCUMENT_EXPLICIT_KEYWORDS):
        return False

    # 2. Check exact conversational match
    if cleaned in GREETING_EXACT_PATTERNS:
        return True

    # 3. Check casual regex patterns
    for pattern in CASUAL_REGEX_PATTERNS:
        if re.search(pattern, cleaned):
            return True

    # 4. Check if query starts with a greeting prefix and contains no heavy request
    words = cleaned.split()
    if words and words[0] in GREETING_PREFIXES:
        remaining_words = [
            w for w in words[1:]
            if w not in ("there", "how", "are", "you", "doing", "today", "friend", "bot", "assistant", "my", "mate")
        ]
        if not remaining_words:
            return True

    return False


TRANSFORMATION_EXACT_PATTERNS = {
    "short this", "shorten this", "shorten it", "shorten that", "short this please", "please short this",
    "make it short", "make it shorter", "make this shorter", "make that shorter",
    "keep it short", "shorter please", "make shorter", "shorter", "tldr", "tl;dr",
    "summarize that", "summarize this", "summarize it", "summarize the above", "summarize",
    "summarize previous", "summarize the previous", "summarize the previous answer", "summarize previous response",
    "give a summary of that", "give a summary of this", "give me a summary of that", "give me a summary of this",
    "brief this", "make it brief", "brief it", "keep it brief",
    "explain simply", "explain it simply", "explain that simply", "explain this simply",
    "explain in simple terms", "explain it in simple terms", "explain that in simple terms",
    "explain simply please", "please explain simply",
    "explain like i'm 5", "explain like im 5", "eli5",
    "simplify this", "simplify that", "simplify it", "make it simple", "make it simpler",
    "in simple words", "in simpler terms", "dumb it down",
    "translate it", "translate this", "translate that", "translate the above",
    "rephrase this", "rephrase that", "rephrase it", "reword this", "reword that", "reword it",
    "rewrite this", "rewrite that", "rewrite it",
    "expand this", "expand that", "expand on this", "expand on that",
    "elaborate on this", "elaborate on that", "elaborate it", "elaborate further",
    "can you elaborate", "more details on that", "more details on this",
    "make it bullet points", "convert to bullet points", "put this in bullet points",
    "format as bullet points", "bullet points please", "in bullet points", "as bullet points"
}

TRANSFORMATION_REGEX_PATTERNS = [
    # Shorten / make shorter / brief
    r"^(?:can you |could you |please |kindly |hey |now )*(?:short|shorten|make (?:it|this|that) (?:short|shorter)|keep it short|make shorter|brief(?: this| that| it)?)(\.|\?|!)*$",
    # Summarize / give summary
    r"^(?:can you |could you |please |kindly |hey |now )*(?:summarize|give (?:me )?(?:a )?summary of)(?: (?:this|that|it|the above|previous(?: answer| response)?))?(\.|\?|!)*$",
    # Explain simply / simplify / eli5
    r"^(?:can you |could you |please |kindly |hey |now )*(?:explain|simplify|make it|put it)(?: (?:this|that|it|the above))?(?: simply| simple| in simple (?:terms|words)| like i'?m 5)(\.|\?|!)*$",
    # Translate it / translate this / translate that [to/into language]
    r"^(?:can you |could you |please |kindly |hey |now )*translate(?: (?:this|that|it|the above))?(?: (?:to|into) [a-zA-Z]+)?(\.|\?|!)*$",
    # Bullet points / list format
    r"^(?:can you |could you |please |kindly |hey |now )*(?:convert|make|put|format)(?: (?:this|that|it|the above))?(?: (?:in|to|as) bullet points?)(?:\.|\?|!)*$",
    # Rephrase / reword / rewrite
    r"^(?:can you |could you |please |kindly |hey |now )*(?:rephrase|reword|rewrite)(?: (?:this|that|it|the above))?(\.|\?|!)*$",
    # Elaborate / expand on this / that
    r"^(?:can you |could you |please |kindly |hey |now )*(?:expand|elaborate)(?: (?:on )?(?:this|that|it|the above|further))?(\.|\?|!)*$",
]


def is_transformation_query(query: str) -> bool:
    """
    Check if a query is a follow-up transformation request targeting the previous assistant response.
    Examples: 'short this', 'summarize that', 'explain simply', 'translate it', 'make it shorter'.
    """
    cleaned = re.sub(r"[^\w\s]", "", query.lower()).strip()
    if not cleaned:
        return False

    # Disqualify if it explicitly targets documents or uploads (these should use RAG)
    doc_markers = {"pdf", "file", "files", "document", "documents", "uploaded", "paper", "textbook"}
    if any(m in cleaned.split() for m in doc_markers) or any(m in query.lower() for m in ("uploaded pdf", "uploaded file", "this pdf", "the pdf", "this document", "the document")):
        return False

    # Check exact match
    if cleaned in TRANSFORMATION_EXACT_PATTERNS:
        return True

    # Check regex match on trimmed query
    q_trimmed = query.strip().lower()
    for pattern in TRANSFORMATION_REGEX_PATTERNS:
        if re.search(pattern, q_trimmed):
            return True

    return False


def detect_intent(query: str) -> str:
    """
    Detect the primary intent of a query.
    Uses heuristic keyword matching for speed — no LLM call.

    Returns one of IntentType values.
    """
    q = query.lower().strip()

    # Calculator: pure math expressions
    from rag.calculator import is_calculator_query
    if is_calculator_query(query):
        return IntentType.CALCULATOR

    # Greeting: lightweight conversational check
    if is_greeting_query(query):
        return IntentType.GREETING

    # Transformation / follow-up check (e.g. "short this", "summarize that")
    if is_transformation_query(query):
        return IntentType.TRANSFORMATION

    # File: PDF-related operations
    file_keywords = [
        "summarize pdf", "summarize file", "read file", "read pdf",
        "uploaded pdf", "uploaded file", "list files", "list uploaded",
        "show uploaded", "which files", "available pdfs",
        "what does this pdf say", "what is in the pdf",
    ]
    file_words = ["pdf", "file", "document", "uploaded"]
    file_actions = ["list", "show", "summarize", "read", "which", "available"]

    if any(kw in q for kw in file_keywords):
        return IntentType.FILE
    if any(w in q for w in file_words) and any(w in q for w in file_actions):
        return IntentType.FILE

    # Image: vision-related
    image_keywords = [
        "what is in this image", "describe this image", "analyze this image",
        "extract text", "ocr", "explain this diagram", "explain this chart",
        "explain this flowchart", "what do you see",
    ]
    if any(kw in q for kw in image_keywords):
        return IntentType.IMAGE

    # Science: math/statistics/plotting
    science_keywords = [
        "solve equation", "derivative", "integral", "matrix",
        "eigenvalue", "plot", "graph", "statistical", "regression",
        "sympy", "numpy", "differential equation", "fourier",
    ]
    if any(kw in q for kw in science_keywords):
        return IntentType.SCIENCE

    # Python: code execution
    python_keywords = [
        "run python", "execute python", "run this code", "python code",
        "write a script", "run code",
    ]
    if any(kw in q for kw in python_keywords):
        return IntentType.PYTHON

    # Default: route to source detection
    return IntentType.GENERAL


# ── Source Router ────────────────────────────────────────────────────────────

def route_sources(query: str) -> dict:
    """
    Intelligent source routing based on query characteristics.
    Merged from both agent_loop.py implementations — uses heuristic
    routing (no LLM call) for speed.

    Returns:
        {primary, fallback, confidence, reasoning, needs_fresh, is_academic}
    """
    if is_transformation_query(query):
        return {
            "primary": "transformation",
            "fallback": "direct_llm",
            "confidence": 1.0,
            "reasoning": "Transformation / follow-up request referencing previous message",
            "needs_fresh": False,
            "is_academic": False,
        }

    query_lower = query.lower()

    # Freshness indicators
    freshness_keywords = [
        "latest", "2025", "2026", "recent", "today", "this week",
        "current", "now", "news", "today's", "breaking", "just", "new",
    ]
    needs_fresh = any(kw in query_lower for kw in freshness_keywords)

    # Academic/Technical indicators
    academic_keywords = [
        "algorithm", "architecture", "framework", "protocol", "theorem",
        "research", "study", "analysis", "design", "implementation",
        "academic", "textbook", "theory", "concept",
    ]
    is_academic = any(kw in query_lower for kw in academic_keywords)

    # Practical/How-to indicators
    practical_keywords = [
        "how to", "tutorial", "guide", "step", "setup", "install",
        "configure", "build", "create", "make",
    ]
    is_practical = any(kw in query_lower for kw in practical_keywords)

    # Decision logic
    if needs_fresh:
        primary, fallback = "web", "pdf"
        confidence = 0.9
        reasoning = "Query requires current information"
    elif is_academic and not needs_fresh:
        primary, fallback = "pdf", "web"
        confidence = 0.85
        reasoning = "Academic/technical query — PDF likely has authoritative sources"
    elif is_practical:
        primary, fallback = "both", "both"
        confidence = 0.75
        reasoning = "Practical query — checking both PDF guides and web resources"
    else:
        primary, fallback = "pdf", "web"
        confidence = 0.6
        reasoning = "General query — starting with PDF, fallback to web"

    return {
        "primary": primary,
        "fallback": fallback,
        "confidence": confidence,
        "reasoning": reasoning,
        "needs_fresh": needs_fresh,
        "is_academic": is_academic,
    }


# ── Complexity Detection ─────────────────────────────────────────────────────

def detect_complexity(query: str) -> dict:
    """
    Analyze query complexity.

    Returns:
        {level: "simple|moderate|complex", score: int, criteria: dict, min_words: int}
    """
    query_lower = query.lower()
    word_count = len(query.split())

    complex_keywords = [
        "compare", "contrast", "analyze", "summarize", "discuss",
        "explain how", "explain why", "what if", "how would",
        "advantages", "disadvantages", "pros and cons",
        "relationship", "impact", "effect",
    ]
    technical_keywords = ["algorithm", "architecture", "framework", "protocol", "implement"]
    multi_part_indicators = [" and ", " or ", ",", ";"]
    sequential_keywords = [" then ", " and then ", " after ", " next "]

    score = 0

    if any(kw in query_lower for kw in complex_keywords):
        score += 2
    if any(kw in query_lower for kw in technical_keywords):
        score += 1

    multi_part_count = sum(query_lower.count(sep) for sep in multi_part_indicators)
    score += min(multi_part_count, 2)

    if any(kw in query_lower for kw in sequential_keywords):
        score += 1

    if word_count > 20:
        score += 1
    elif word_count < 5:
        score -= 1

    if score >= 4:
        level = "complex"
        criteria = {"min_words": 10, "description": "Complex — detailed, multi-faceted answer", "strict": False}
    elif score >= 2:
        level = "moderate"
        criteria = {"min_words": 7, "description": "Moderate — balanced answer", "strict": False}
    else:
        level = "simple"
        criteria = {"min_words": 5, "description": "Simple — concise answer acceptable", "strict": True}

    logger.info(f"📊 Complexity: {level} (score={score}) | min_words={criteria['min_words']}")

    return {
        "level": level,
        "score": score,
        "criteria": criteria,
        "min_words": criteria["min_words"],
    }


# ── Task Decomposition ──────────────────────────────────────────────────────

def should_decompose(query: str) -> bool:
    """Quick check: does this query need decomposition?"""
    complexity = detect_complexity(query)
    if complexity["level"] != "complex":
        # Exception: freshness + explanation pattern
        ql = query.lower()
        freshness = ["latest", "recent", "new", "current"]
        explain = ["explain", "describe", "understand", "how", "what's"]
        if any(f in ql for f in freshness) and " and " in ql and any(e in ql for e in explain):
            return True
        return False
    return True


def decompose_query(query: str) -> dict:
    """
    Analyze if query should be decomposed into sub-tasks.

    Returns:
        {should_decompose, subtasks, reasoning, synthesis_instruction}
    """
    if not should_decompose(query):
        return {
            "should_decompose": False,
            "subtasks": [],
            "reasoning": f"Single-step execution sufficient",
        }

    query_lower = query.lower()
    has_and = " and " in query_lower or ", " in query_lower

    comparison_words = ["compare", "contrast", "difference", "vs", "versus"]
    has_comparison = any(w in query_lower for w in comparison_words)

    freshness = ["latest", "recent", "new", "current"]
    explain = ["explain", "describe", "understand", "how", "what"]
    has_freshness_explain = (
        any(f in query_lower for f in freshness)
        and has_and
        and any(e in query_lower for e in explain)
    )

    decomp_indicators = [
        has_comparison and has_and,
        query_lower.count(",") >= 2,
        ("explain" in query_lower or "discuss" in query_lower) and has_and,
        "pros and cons" in query_lower or "benefits and drawbacks" in query_lower,
        has_freshness_explain,
    ]

    if not any(decomp_indicators):
        return {
            "should_decompose": False,
            "subtasks": [],
            "reasoning": "Complex but no multi-part indicators detected",
        }

    # LLM-based decomposition
    prompt = f"""Analyze this query and break it into focused sub-tasks if needed.

Query: {query}

Return valid JSON ONLY (no markdown, no extra text):{{
  "can_decompose": true,
  "subtasks": [
    {{"number": 1, "task": "first focused question", "purpose": "what this explores"}},
    {{"number": 2, "task": "second focused question", "purpose": "what this explores"}}
  ],
  "synthesis_instruction": "How to combine all results into a coherent answer"
}}

Or if not decomposable:{{
  "can_decompose": false,
  "subtasks": [],
  "synthesis_instruction": ""
}}"""

    try:
        raw = call_llm(prompt, max_tokens=512)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            json_str = json_match.group().replace('\\n', ' ').replace('\\', '')
            decomp = json.loads(json_str)

            if decomp.get("can_decompose") and decomp.get("subtasks"):
                logger.info(f"🔀 Decomposed into {len(decomp['subtasks'])} sub-tasks")
                return {
                    "should_decompose": True,
                    "subtasks": decomp["subtasks"],
                    "reasoning": f"Multi-part query: {len(decomp['subtasks'])} focused sub-tasks",
                    "synthesis_instruction": decomp.get("synthesis_instruction", "Combine results coherently"),
                }
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Decomposition JSON parse failed: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Decomposition LLM failed: {e}")

    return {
        "should_decompose": False,
        "subtasks": [],
        "reasoning": "Decomposition analysis inconclusive — single-step",
    }


# ── Main Planner ─────────────────────────────────────────────────────────────

class Planner:
    """
    Unified planning engine.
    Produces a complete execution plan from a user query.
    """

    def plan(self, query: str) -> dict:
        """
        Produce a full execution plan.

        Returns:
            {
                intent: str,
                source: str,
                route_info: dict,
                complexity: dict,
                decomposition: dict,
                needs_rewrite: bool,
                reasoning_depth: str,
                max_retries: int,
            }
        """
        # Step 1: Detect intent
        intent = detect_intent(query)
        logger.info(f"🎯 Intent: {intent}")

        # Short-circuit for greeting intent
        if intent == IntentType.GREETING:
            return {
                "intent": intent,
                "source": "direct_llm",
                "route_info": {"primary": "direct_llm", "fallback": "direct_llm", "confidence": 1.0, "reasoning": "Greeting / conversational message"},
                "complexity": {"level": "simple", "score": 0, "min_words": 5},
                "decomposition": {"should_decompose": False, "subtasks": []},
                "needs_rewrite": False,
                "reasoning_depth": "simple",
                "max_retries": 1,
            }

        # Short-circuit for transformation intent
        if intent == IntentType.TRANSFORMATION:
            return {
                "intent": intent,
                "source": "transformation",
                "route_info": {"primary": "transformation", "fallback": "direct_llm", "confidence": 1.0, "reasoning": "Follow-up / transformation of previous response"},
                "complexity": {"level": "simple", "score": 0, "min_words": 5},
                "decomposition": {"should_decompose": False, "subtasks": []},
                "needs_rewrite": False,
                "reasoning_depth": "simple",
                "max_retries": 1,
            }

        # Short-circuit for tool-specific intents
        if intent in (IntentType.CALCULATOR, IntentType.FILE, IntentType.IMAGE,
                       IntentType.SCIENCE, IntentType.PYTHON):
            return {
                "intent": intent,
                "source": intent,
                "route_info": {"primary": intent, "fallback": "web", "confidence": 0.95, "reasoning": f"Direct {intent} tool"},
                "complexity": {"level": "simple", "score": 0, "min_words": 5},
                "decomposition": {"should_decompose": False, "subtasks": []},
                "needs_rewrite": False,
                "reasoning_depth": "simple",
                "max_retries": 1,
            }

        # Step 2: Route sources
        route_info = route_sources(query)

        # Step 3: Detect complexity
        complexity = detect_complexity(query)

        # Step 4: Check decomposition
        decomposition = decompose_query(query)

        # Step 5: Determine retry strategy from routing confidence
        confidence = route_info["confidence"]
        if confidence >= 0.85:
            max_retries = 1
        elif confidence >= 0.75:
            max_retries = 2
        else:
            max_retries = settings.AGENT_MAX_RETRIES

        plan = {
            "intent": intent,
            "source": route_info["primary"],
            "route_info": route_info,
            "complexity": complexity,
            "decomposition": decomposition,
            "needs_rewrite": complexity["level"] != "simple",
            "reasoning_depth": "deep" if complexity["level"] == "complex" else "simple",
            "max_retries": max_retries,
        }

        logger.info(f"📋 Plan: source={plan['source']} | complexity={complexity['level']} | "
                     f"decompose={decomposition.get('should_decompose', False)} | retries={max_retries}")

        return plan


def plan_query(query: str, memory_context: str = "") -> AgentPlan:
    """Helper function to run planner and return structured AgentPlan."""
    planner = Planner()
    raw = planner.plan(query)
    intent = raw.get("intent", "general")
    complexity = raw.get("complexity", {}).get("level", "simple")
    search_type = raw.get("source", "direct_llm")

    if intent == IntentType.GREETING:
        recommended_tool = "direct_llm"
        search_type = "direct_llm"
    elif intent == IntentType.TRANSFORMATION:
        recommended_tool = "transformation"
        search_type = "transformation"
    else:
        from tools.registry import tool_registry
        recommended_tool = tool_registry.select_tool(intent, query)
        if not recommended_tool:
            if search_type in ("web", "pdf", "both"):
                recommended_tool = search_type
            else:
                recommended_tool = "direct_llm"

    subtasks = raw.get("decomposition", {}).get("subtasks", [])
    if not subtasks:
        subtasks = [
            {"step_num": 1, "tool": recommended_tool, "description": query, "query": query}
        ]

    return AgentPlan(
        original_query=query,
        intent=intent,
        complexity_level=complexity,
        search_type=search_type,
        recommended_tool=recommended_tool,
        subtasks=subtasks,
        needs_rewrite=raw.get("needs_rewrite", False),
        max_retries=raw.get("max_retries", 1),
        raw_plan_dict=raw
    )
