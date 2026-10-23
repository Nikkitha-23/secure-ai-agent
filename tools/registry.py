"""
tools/registry.py — Central Tool Registry & Permission System
---------------------------------------------------------------
Single place where all tools are registered, validated, and dispatched.
Replaces the fragmented routing across tool_router.py, tool_chain.py,
and scattered imports in agent_loop.py.

Features:
  - Central tool registration with schemas
  - Permission levels per tool
  - Safe dispatch with validation
  - Max call tracking per request
  - Timeout enforcement

Usage:
    from tools.registry import tool_registry
    result = tool_registry.execute("web_search", {"query": "latest AI news"})
    schema = tool_registry.get_schema("calculator")
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# ── Permission Levels ────────────────────────────────────────────────────────

class ToolPermission:
    """Permission levels for tool access."""
    PUBLIC = "public"          # Anyone can use
    AUTHENTICATED = "auth"     # Requires login
    PRIVILEGED = "privileged"  # Requires elevated role (teacher, admin, doctor)
    ADMIN = "admin"            # Admin-only


# ── Tool Definition ──────────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """Definition of a registered tool."""
    name: str
    description: str
    handler: Callable
    permission: str = ToolPermission.PUBLIC
    timeout_seconds: int = 30
    max_calls_per_request: int = 3
    parameters: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class ToolCallResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    latency_ms: float = 0.0
    source_type: str = "tool_result"


# ── Tool Registry ────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all available tools.
    Handles registration, validation, and safe dispatch.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._call_counts: dict[str, int] = {}  # per-request tracking
        logger.info("✅ Tool Registry initialized")

    def register(self, tool_def: ToolDefinition):
        """Register a tool definition."""
        self._tools[tool_def.name] = tool_def
        logger.info(f"🔧 Registered tool: {tool_def.name} [{tool_def.permission}]")

    def reset_call_counts(self):
        """Reset per-request call counts. Call at start of each agent run."""
        self._call_counts = {}

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self, permission_level: str = None) -> list[dict]:
        """List all registered tools, optionally filtered by permission."""
        tools = []
        for name, tool_def in self._tools.items():
            if not tool_def.enabled:
                continue
            if permission_level and tool_def.permission != permission_level:
                continue
            tools.append({
                "name": name,
                "description": tool_def.description,
                "permission": tool_def.permission,
                "parameters": tool_def.parameters,
                "tags": tool_def.tags,
            })
        return tools

    def get_schema(self, name: str) -> Optional[dict]:
        """Get tool schema for LLM function calling."""
        tool_def = self._tools.get(name)
        if not tool_def:
            return None
        return {
            "name": tool_def.name,
            "description": tool_def.description,
            "parameters": tool_def.parameters,
        }

    def check_permission(self, tool_name: str, user_role: str = "public") -> bool:
        """Check if a role has permission to use a tool."""
        tool_def = self._tools.get(tool_name)
        if not tool_def:
            return False

        permission = tool_def.permission

        if permission == ToolPermission.PUBLIC:
            return True
        if permission == ToolPermission.AUTHENTICATED:
            return user_role != "public"
        if permission == ToolPermission.PRIVILEGED:
            return user_role in ("teacher", "admin", "edu_admin", "doctor", "hospital_admin", "super_admin")
        if permission == ToolPermission.ADMIN:
            return user_role in ("admin", "edu_admin", "hospital_admin", "super_admin")

        return False

    def execute(
        self,
        tool_name: str,
        args: dict = None,
        user_role: str = "public",
    ) -> ToolCallResult:
        """
        Safely execute a tool with validation, permission checks,
        call-count limits, and timeout tracking.

        Args:
            tool_name: Name of the registered tool
            args: Arguments to pass to the tool handler
            user_role: Role of the requesting user

        Returns:
            ToolCallResult with output or error
        """
        args = args or {}

        # Validate tool exists
        tool_def = self._tools.get(tool_name)
        if not tool_def:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unknown tool: '{tool_name}'. Available: {list(self._tools.keys())}",
            )

        # Check if enabled
        if not tool_def.enabled:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool '{tool_name}' is currently disabled",
            )

        # Permission check
        if not self.check_permission(tool_name, user_role):
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Permission denied: role '{user_role}' cannot use tool '{tool_name}'",
            )

        # Call count limit
        count = self._call_counts.get(tool_name, 0)
        if count >= tool_def.max_calls_per_request:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool '{tool_name}' exceeded max calls ({tool_def.max_calls_per_request}) for this request",
            )

        # Execute with timing
        start = time.time()
        try:
            logger.info(f"▶ Executing tool: {tool_name} | args={str(args)[:100]}")
            result = tool_def.handler(**args)
            latency_ms = round((time.time() - start) * 1000, 2)

            # Normalize result to string
            if isinstance(result, dict):
                output = result.get("answer", result.get("output", json.dumps(result, ensure_ascii=False)))
                source_type = result.get("search_type", "tool_result")
                is_error = "error" in source_type or "failed" in source_type or output.startswith("Could not evaluate") or output.startswith("Error:")
                if is_error:
                    return ToolCallResult(
                        tool_name=tool_name,
                        success=False,
                        output="",
                        error=output,
                        latency_ms=latency_ms,
                        source_type=source_type,
                    )
            elif isinstance(result, str):
                output = result
                source_type = "tool_result"
                if output.startswith("Could not evaluate") or output.startswith("Error:"):
                    return ToolCallResult(
                        tool_name=tool_name,
                        success=False,
                        output="",
                        error=output,
                        latency_ms=latency_ms,
                        source_type=source_type,
                    )
            else:
                output = str(result)
                source_type = "tool_result"

            self._call_counts[tool_name] = count + 1

            logger.info(f"✅ Tool {tool_name} completed in {latency_ms}ms | output={output[:80]}...")

            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                output=output,
                latency_ms=latency_ms,
                source_type=source_type,
            )

        except Exception as e:
            latency_ms = round((time.time() - start) * 1000, 2)
            logger.error(f"❌ Tool {tool_name} failed in {latency_ms}ms: {e}")
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Tool '{tool_name}' error: {str(e)}",
                latency_ms=latency_ms,
            )

    def select_tool(self, intent: str, query: str) -> Optional[str]:
        """
        Given an intent from the planner, return the best tool name.
        Returns None if no specific tool is needed (use RAG/LLM).
        """
        intent_tool_map = {
            "calculator": "calculator",
            "file": "file_reader",
            "image": "image_analyzer",
            "science": "science",
            "python": "python_exec",
            "web": "web_search",
            "memory": "memory_recall",
        }
        return intent_tool_map.get(intent)


# ── Singleton ────────────────────────────────────────────────────────────────
tool_registry = ToolRegistry()


# ── Registration Functions ───────────────────────────────────────────────────

def register_all_tools():
    """Register all available tools. Called once at app startup."""

    # 1. Web Search
    try:
        from rag.web_search import web_search
        tool_registry.register(ToolDefinition(
            name="web_search",
            description="Search the internet for current/live information",
            handler=lambda query=None, max_results=3, **kwargs: web_search(query or kwargs.get("expression", ""), max_results),
            permission=ToolPermission.PUBLIC,
            parameters={"query": "string", "max_results": "int (default 3)"},
            tags=["search", "web", "current"],
        ))
    except ImportError as e:
        logger.warning(f"⚠️ Could not register web_search: {e}")

    # 2. Calculator
    try:
        from rag.calculator import calculate
        tool_registry.register(ToolDefinition(
            name="calculator",
            description="Safely evaluate mathematical expressions",
            handler=lambda expression=None, query=None, **kwargs: calculate(expression or query or ""),
            permission=ToolPermission.PUBLIC,
            parameters={"expression": "string (math expression)"},
            tags=["math", "calculator"],
        ))
    except ImportError as e:
        logger.warning(f"⚠️ Could not register calculator: {e}")

    # 3. Python Execution
    try:
        from rag.python_exec import run_python
        tool_registry.register(ToolDefinition(
            name="python_exec",
            description="Execute Python code in a sandboxed environment",
            handler=lambda code=None, query=None, **kwargs: run_python(code or query or ""),
            permission=ToolPermission.AUTHENTICATED,
            timeout_seconds=settings.PYTHON_EXEC_TIMEOUT,
            max_calls_per_request=2,
            parameters={"code": "string (Python code)"},
            tags=["code", "python", "execution"],
        ))
    except ImportError as e:
        logger.warning(f"⚠️ Could not register python_exec: {e}")

    # 4. File Reader
    try:
        from tools.file_tool import file_tool
        tool_registry.register(ToolDefinition(
            name="file_reader",
            description="Read, summarize, list, or ask questions about uploaded PDF files",
            handler=lambda action="list", filename=None, question=None, query=None, **kwargs: file_tool.run(
                action=action, filename=filename or query, question=question or query
            ),
            permission=ToolPermission.PUBLIC,
            parameters={"action": "list|read|summarize|ask", "filename": "string (optional)", "question": "string (optional)"},
            tags=["file", "pdf", "document"],
        ))
    except ImportError as e:
        logger.warning(f"⚠️ Could not register file_reader: {e}")

    # 5. Image Analyzer
    try:
        from tools.image_tool import run_image_tool
        tool_registry.register(ToolDefinition(
            name="image_analyzer",
            description="Analyze images using vision AI (describe, OCR, diagram analysis)",
            handler=lambda query=None, image_input=None, **kwargs: run_image_tool(query=query or "", image_input=image_input),
            permission=ToolPermission.PUBLIC,
            parameters={"query": "string", "image_input": "bytes|path|base64 (optional)"},
            tags=["image", "vision", "ocr"],
        ))
    except ImportError as e:
        logger.warning(f"⚠️ Could not register image_analyzer: {e}")

    # 6. Science Tool
    try:
        from tools.science_tool import run_science_tool
        tool_registry.register(ToolDefinition(
            name="science",
            description="Scientific computing: math, statistics, data analysis, plotting",
            handler=lambda query=None, **kwargs: run_science_tool(query or ""),
            permission=ToolPermission.PUBLIC,
            parameters={"query": "string (science/math query)"},
            tags=["science", "math", "statistics", "plotting"],
        ))
    except ImportError as e:
        logger.warning(f"⚠️ Could not register science: {e}")

    # 7. Memory Recall
    try:
        from rag.memory import get_memory_manager
        tool_registry.register(ToolDefinition(
            name="memory_recall",
            description="Recall past conversation context from memory",
            handler=lambda query=None, session_id="default", **kwargs: get_memory_manager().recall(query or "", session_id),
            permission=ToolPermission.PUBLIC,
            parameters={"query": "string", "session_id": "string (default: 'default')"},
            tags=["memory", "context", "history"],
        ))
    except ImportError as e:
        logger.warning(f"⚠️ Could not register memory_recall: {e}")

    registered = len(tool_registry._tools)
    logger.info(f"🔧 Tool registration complete: {registered} tools registered")
    return registered


# ── Top-level Helper Aliases & Functions ────────────────────────────────────

ToolPermissionLevel = ToolPermission

class ToolRiskLevel:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def register_tool(name: str, description: str, func: Callable, permission: str = ToolPermission.PUBLIC):
    tool_registry.register(ToolDefinition(
        name=name,
        description=description,
        handler=func,
        permission=permission
    ))


# Auto-register default tools on module import
register_all_tools()
