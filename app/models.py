"""
app/models.py — Pydantic Request/Response Models
--------------------------------------------------
Shared data models for the FastAPI application.
All endpoints use these for validation and serialization.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class SearchType(str, Enum):
    PDF = "pdf"
    WEB = "web"
    BOTH = "both"
    CALCULATOR = "calculator"
    FILE = "file"
    SCIENCE = "science"
    IMAGE = "image"
    PYTHON = "python"
    TOOL_CHAIN = "tool_chain"
    MULTI_STEP = "multi-step"
    CACHE = "cache"
    BLOCKED = "blocked"
    ERROR = "error"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    """Distinguishes where evidence came from."""
    RETRIEVED = "retrieved"        # From vector/BM25 retrieval
    WEB = "web"                    # From web search
    CALCULATED = "calculated"      # From calculator/python execution
    TOOL_RESULT = "tool_result"    # From a tool (file, image, science)
    MODEL_KNOWLEDGE = "model"      # From LLM's own knowledge
    MEMORY = "memory"              # From memory recall


# ── Request Models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Standard query request."""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: str = Field(default="default", description="Session identifier")


class AgentRequest(BaseModel):
    """Unified core agent request."""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_role: Optional[str] = Field(default="student", description="User role for permission checks")
    user_id: Optional[str] = Field(default="default_user", description="User ID for memory partition")


class UploadRequest(BaseModel):
    """File upload metadata."""
    filename: str
    content_type: str = "application/pdf"


# ── Response Models ──────────────────────────────────────────────────────────

class GroundingInfo(BaseModel):
    """Hallucination grounding result."""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = Field(default="unknown")  # grounded | partial | hallucinated
    detail: str = Field(default="")


class ReflectionInfo(BaseModel):
    """Agent reflection result."""
    good: bool = True
    reason: str = ""
    retry_source: str = ""
    complexity: str = "simple"


class PlanInfo(BaseModel):
    """Agent planning result."""
    source: str = "pdf"
    needs_rewrite: bool = False
    reasoning_depth: str = "simple"
    route_info: dict = Field(default_factory=dict)


class StepLog(BaseModel):
    """Step execution log."""
    step_num: int
    tool: str
    purpose: str
    status: str = "done"
    result: str = ""
    duration_ms: int = 0


class SourceInfo(BaseModel):
    """Source citation with type tagging."""
    name: str
    source_type: SourceType = SourceType.RETRIEVED
    relevance_score: float = 0.0


class AgentResponse(BaseModel):
    """Full agent response with all metadata."""
    answer: str
    sources: List[str] = Field(default_factory=list)
    source_details: List[SourceInfo] = Field(default_factory=list)
    search_type: str = "unknown"
    attempts: int = 1
    is_multi_step: bool = False
    plan: Dict[str, Any] = Field(default_factory=dict)
    reflection: Dict[str, Any] = Field(default_factory=dict)
    grounding: Dict[str, Any] = Field(default_factory=dict)
    trace: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    steps: List[StepLog] = Field(default_factory=list)
    subtask_results: List[Dict[str, Any]] = Field(default_factory=list)
    cached: bool = False

    class Config:
        extra = "allow"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "2.0.0"
    warnings: List[str] = Field(default_factory=list)
