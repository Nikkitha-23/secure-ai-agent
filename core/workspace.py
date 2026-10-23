"""
core/workspace.py — Agent Workspace & Task State
-------------------------------------------------
Tracks short-term task state, active sub-tasks, execution steps,
tool outputs, retry counts, and scratchpad notes for multi-step reasoning.

Durable persistence is backed by SQLite (data/workspace.db) with
ACID transactions, ensuring task state survives server restarts.
"""

import os
import time
import uuid
import json
import sqlite3
import logging
import threading
import contextlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

WORKSPACE_DIR = os.path.join("data", "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
DEFAULT_DB_PATH = os.path.join("data", "workspace.db")

MAX_STORED_OUTPUT_LEN = 2000


def truncate_output(text: Any, max_len: int = MAX_STORED_OUTPUT_LEN) -> str:
    """Limit stored output sizes to prevent task bloat."""
    if text is None:
        return ""
    str_val = str(text)
    if len(str_val) <= max_len:
        return str_val
    truncated_count = len(str_val) - max_len
    return str_val[:max_len] + f"... [truncated {truncated_count} chars]"


class CallableList(list):
    """
    List that can be accessed both as an iterable attribute
    (ws.completed_steps) and as a callable (ws.completed_steps())
    for full backward compatibility.
    """
    def __call__(self):
        return self


class _CallableStepHelper:
    """
    Helper object wrapping Optional[WorkspaceStep] that allows both
    ws.current_step (attribute access) and ws.current_step() (callable)
    to work seamlessly.
    """
    def __init__(self, step: Optional['WorkspaceStep']):
        self._step = step

    def __call__(self) -> Optional['WorkspaceStep']:
        return self._step

    def __getattr__(self, item):
        if self._step is not None:
            return getattr(self._step, item)
        raise AttributeError(f"'NoneType' object has no attribute '{item}'")

    def __bool__(self):
        return self._step is not None

    def __repr__(self):
        return repr(self._step)

    def to_dict(self) -> Optional[dict]:
        return self._step.to_dict() if self._step else None


# ── Step State ────────────────────────────────────────────────────────────────

@dataclass
class WorkspaceStep:
    step_num: int
    tool: str
    purpose: str
    status: str = "pending"       # pending | running | completed | failed | waiting | done
    result: str = ""
    error: str = ""
    retry_count: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: int = 0

    def start(self):
        self.status = "running"
        if not self.started_at:
            self.started_at = time.time()

    def complete(self, result: str):
        self.status = "completed"
        self.result = truncate_output(result)
        self.completed_at = time.time()
        start = self.started_at or self.completed_at
        self.duration_ms = max(0, int((self.completed_at - start) * 1000))

    def fail(self, error: str):
        self.status = "failed"
        self.error = str(error)
        self.result = f"Error: {self.error}"
        self.completed_at = time.time()
        start = self.started_at or self.completed_at
        self.duration_ms = max(0, int((self.completed_at - start) * 1000))

    def to_dict(self) -> dict:
        return {
            "step_num": self.step_num,
            "tool": self.tool,
            "purpose": self.purpose,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WorkspaceStep':
        return cls(
            step_num=data.get("step_num", 1),
            tool=data.get("tool", ""),
            purpose=data.get("purpose", ""),
            status=data.get("status", "pending"),
            result=data.get("result", ""),
            error=data.get("error", ""),
            retry_count=data.get("retry_count", 0),
            started_at=data.get("started_at", 0.0),
            completed_at=data.get("completed_at", 0.0),
            duration_ms=data.get("duration_ms", 0)
        )


# ── Agent Workspace ───────────────────────────────────────────────────────────

@dataclass
class AgentWorkspace:
    """
    Per-task workspace tracking state, execution plan, subtasks, steps,
    results, and errors.
    """
    session_id: str = field(default_factory=lambda: f"sess_{str(uuid.uuid4())[:8]}")
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_query: str = ""
    status: str = "pending"       # pending | running | completed | failed (also accepts done, idle)
    plan: Dict[str, Any] = field(default_factory=dict)
    steps: List[WorkspaceStep] = field(default_factory=list)
    final_answer: str = ""
    error: str = ""
    retry_count: int = 0
    scratchpad: List[Dict[str, Any]] = field(default_factory=list)
    search_type: str = ""
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    total_duration_ms: int = 0
    _storage: Any = field(default=None, repr=False, compare=False)

    # ── Step Accessors (Callable + Property) ──────────────────────────────────

    @property
    def current_step(self) -> _CallableStepHelper:
        for s in self.steps:
            if s.status in ["pending", "waiting", "running"]:
                return _CallableStepHelper(s)
        return _CallableStepHelper(None)

    @property
    def completed_steps(self) -> CallableList:
        return CallableList([s for s in self.steps if s.status in ["completed", "done"]])

    @property
    def pending_steps(self) -> CallableList:
        return CallableList([s for s in self.steps if s.status in ["pending", "waiting"]])

    @property
    def failed_steps(self) -> CallableList:
        return CallableList([s for s in self.steps if s.status == "failed"])

    # ── Step & State Management ───────────────────────────────────────────────

    def add_step(self, tool: str, purpose: str, step_num: Optional[int] = None) -> WorkspaceStep:
        num = step_num if step_num is not None else len(self.steps) + 1
        step = WorkspaceStep(
            step_num=num,
            tool=tool,
            purpose=purpose,
            status="pending"
        )
        self.steps.append(step)
        self.updated_at = time.time()
        self.save()
        return step

    def start(self):
        self.status = "running"
        self.updated_at = time.time()
        self.save()

    def note(self, text: str):
        """Add a scratchpad note."""
        entry = {
            "time": time.time(),
            "note": text
        }
        self.scratchpad.append(entry)
        self.updated_at = time.time()
        logger.debug(f"📝 Scratchpad: {text}")

    def complete(self, answer: str, search_type: str = "", confidence: float = 0.0):
        # Do not mark completed if required steps failed
        if len(self.failed_steps) > 0:
            logger.warning(f"Task {self.task_id} has failed steps; recording failure instead of complete.")
            self.fail(f"{len(self.failed_steps)} step(s) failed during execution.")
            return

        self.status = "completed"
        self.final_answer = answer
        self.search_type = search_type or self.search_type
        self.confidence = confidence or self.confidence
        self.completed_at = time.time()
        self.updated_at = self.completed_at
        self.total_duration_ms = max(0, int((self.completed_at - self.created_at) * 1000))
        self.save()

    def fail(self, reason: str):
        self.status = "failed"
        self.error = str(reason)
        self.completed_at = time.time()
        self.updated_at = self.completed_at
        self.total_duration_ms = max(0, int((self.completed_at - self.created_at) * 1000))
        self.note(f"Task failed: {reason}")
        self.save()

    def summary(self) -> dict:
        step_statuses = []
        for s in self.steps:
            icon = {
                "pending": "⏳",
                "waiting": "⏳",
                "running": "🔄",
                "completed": "✅",
                "done": "✅",
                "failed": "❌"
            }.get(s.status, "❓")
            step_statuses.append({
                "step": s.step_num,
                "tool": s.tool,
                "purpose": s.purpose,
                "status": s.status,
                "icon": icon,
                "duration_ms": s.duration_ms,
                "error": s.error
            })

        curr = self.current_step()
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "status": self.status,
            "query": self.original_query[:100],
            "original_query": self.original_query,
            "current_step": curr.step_num if curr else None,
            "steps": step_statuses,
            "completed": len(self.completed_steps),
            "pending": len(self.pending_steps),
            "failed": len(self.failed_steps),
            "total": len(self.steps),
            "retry_count": self.retry_count,
            "search_type": self.search_type,
            "confidence": self.confidence,
            "duration_ms": self.total_duration_ms,
            "error": self.error,
            "final_answer": self.final_answer,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "scratchpad_entries": len(self.scratchpad)
        }

    def to_dict(self) -> dict:
        curr = self.current_step()
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "original_query": self.original_query,
            "status": self.status,
            "plan": self.plan,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": curr.to_dict() if curr else None,
            "completed_steps": [s.to_dict() for s in self.completed_steps],
            "pending_steps": [s.to_dict() for s in self.pending_steps],
            "failed_steps": [s.to_dict() for s in self.failed_steps],
            "scratchpad": self.scratchpad,
            "final_answer": self.final_answer,
            "error": self.error,
            "retry_count": self.retry_count,
            "search_type": self.search_type,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms
        }

    @classmethod
    def from_dict(cls, data: dict, storage: Any = None) -> 'AgentWorkspace':
        raw_steps = data.get("steps", [])
        steps = [WorkspaceStep.from_dict(s) if isinstance(s, dict) else s for s in raw_steps]
        ws = cls(
            session_id=data.get("session_id", ""),
            task_id=data.get("task_id", ""),
            original_query=data.get("original_query") or data.get("query") or "",
            status=data.get("status", "pending"),
            plan=data.get("plan", {}),
            steps=steps,
            final_answer=data.get("final_answer", ""),
            error=data.get("error", ""),
            retry_count=data.get("retry_count", 0),
            scratchpad=data.get("scratchpad", []),
            search_type=data.get("search_type", ""),
            confidence=data.get("confidence", 0.0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            completed_at=data.get("completed_at", 0.0),
            total_duration_ms=data.get("total_duration_ms", 0),
            _storage=storage
        )
        return ws

    def save(self):
        """Persist workspace state to durable storage and fallback JSON."""
        try:
            # 1. Save to configured or global SQLite storage
            storage = self._storage or get_workspace_storage()
            storage.save(self)
        except Exception as e:
            logger.warning(f"Failed to save workspace {self.task_id} to SQLite: {e}")

        # 2. Also save to JSON file as supplementary record
        try:
            filepath = os.path.join(WORKSPACE_DIR, f"{self.task_id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as ex:
            logger.debug(f"JSON backup write skipped for {self.task_id}: {ex}")


# ── Storage Interface & SQLite Implementation ────────────────────────────────

class WorkspaceStorage:
    """Abstract interface for workspace persistence."""
    def save(self, workspace: AgentWorkspace) -> None:
        raise NotImplementedError

    def get(self, task_id: str) -> Optional[AgentWorkspace]:
        raise NotImplementedError

    def get_by_session(self, session_id: str, limit: int = 50) -> List[AgentWorkspace]:
        raise NotImplementedError

    def delete(self, task_id: str) -> bool:
        raise NotImplementedError

    def clear_session(self, session_id: str) -> int:
        raise NotImplementedError


class SQLiteWorkspaceStorage(WorkspaceStorage):
    """
    SQLite-backed durable workspace store.
    Thread-safe with connection management, automatic connection closure,
    and automatic schema migration.
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextlib.contextmanager
    def _connection(self):
        """Context manager that ensures the SQLite connection is closed immediately."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._lock:
            with self._connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workspaces (
                        task_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        original_query TEXT NOT NULL,
                        status TEXT NOT NULL,
                        plan_json TEXT NOT NULL DEFAULT '{}',
                        steps_json TEXT NOT NULL DEFAULT '[]',
                        scratchpad_json TEXT NOT NULL DEFAULT '[]',
                        final_answer TEXT NOT NULL DEFAULT '',
                        error TEXT NOT NULL DEFAULT '',
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        search_type TEXT NOT NULL DEFAULT '',
                        confidence REAL NOT NULL DEFAULT 0.0,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        completed_at REAL NOT NULL DEFAULT 0.0,
                        total_duration_ms INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_session ON workspaces(session_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_status ON workspaces(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_updated ON workspaces(updated_at DESC);")

    def save(self, workspace: AgentWorkspace) -> None:
        with self._lock:
            with self._connection() as conn:
                steps_data = [s.to_dict() for s in workspace.steps]
                conn.execute("""
                    INSERT INTO workspaces (
                        task_id, session_id, original_query, status,
                        plan_json, steps_json, scratchpad_json,
                        final_answer, error, retry_count,
                        search_type, confidence,
                        created_at, updated_at, completed_at, total_duration_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        session_id = excluded.session_id,
                        original_query = excluded.original_query,
                        status = excluded.status,
                        plan_json = excluded.plan_json,
                        steps_json = excluded.steps_json,
                        scratchpad_json = excluded.scratchpad_json,
                        final_answer = excluded.final_answer,
                        error = excluded.error,
                        retry_count = excluded.retry_count,
                        search_type = excluded.search_type,
                        confidence = excluded.confidence,
                        updated_at = excluded.updated_at,
                        completed_at = excluded.completed_at,
                        total_duration_ms = excluded.total_duration_ms;
                """, (
                    workspace.task_id,
                    workspace.session_id,
                    workspace.original_query,
                    workspace.status,
                    json.dumps(workspace.plan),
                    json.dumps(steps_data),
                    json.dumps(workspace.scratchpad),
                    workspace.final_answer,
                    workspace.error,
                    workspace.retry_count,
                    workspace.search_type,
                    workspace.confidence,
                    workspace.created_at,
                    workspace.updated_at,
                    workspace.completed_at,
                    workspace.total_duration_ms
                ))

    def get(self, task_id: str) -> Optional[AgentWorkspace]:
        if not task_id or not isinstance(task_id, str):
            return None
        with self._lock:
            with self._connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM workspaces WHERE task_id = ?",
                    (task_id.strip(),)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return self._row_to_workspace(row)

    def get_by_session(self, session_id: str, limit: int = 50) -> List[AgentWorkspace]:
        if not session_id:
            return []
        with self._lock:
            with self._connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM workspaces WHERE session_id = ? ORDER BY updated_at DESC LIMIT ?",
                    (session_id.strip(), limit)
                )
                rows = cursor.fetchall()
                return [self._row_to_workspace(r) for r in rows]

    def delete(self, task_id: str) -> bool:
        with self._lock:
            with self._connection() as conn:
                cursor = conn.execute("DELETE FROM workspaces WHERE task_id = ?", (task_id,))
                return cursor.rowcount > 0

    def clear_session(self, session_id: str) -> int:
        with self._lock:
            with self._connection() as conn:
                cursor = conn.execute("DELETE FROM workspaces WHERE session_id = ?", (session_id,))
                return cursor.rowcount

    def all_tasks(self, limit: int = 100) -> List[AgentWorkspace]:
        with self._lock:
            with self._connection() as conn:
                cursor = conn.execute("SELECT * FROM workspaces ORDER BY updated_at DESC LIMIT ?", (limit,))
                rows = cursor.fetchall()
                return [self._row_to_workspace(r) for r in rows]

    def _row_to_workspace(self, row: sqlite3.Row) -> AgentWorkspace:
        try:
            plan = json.loads(row["plan_json"])
        except Exception:
            plan = {}
        try:
            raw_steps = json.loads(row["steps_json"])
            steps = [WorkspaceStep.from_dict(s) for s in raw_steps]
        except Exception:
            steps = []
        try:
            scratchpad = json.loads(row["scratchpad_json"])
        except Exception:
            scratchpad = []

        return AgentWorkspace(
            task_id=row["task_id"],
            session_id=row["session_id"],
            original_query=row["original_query"],
            status=row["status"],
            plan=plan,
            steps=steps,
            scratchpad=scratchpad,
            final_answer=row["final_answer"],
            error=row["error"],
            retry_count=row["retry_count"],
            search_type=row["search_type"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            total_duration_ms=row["total_duration_ms"],
            _storage=self
        )


# ── Storage Singleton ─────────────────────────────────────────────────────────

_storage_instance: Optional[SQLiteWorkspaceStorage] = None

def get_workspace_storage(db_path: Optional[str] = None) -> SQLiteWorkspaceStorage:
    global _storage_instance
    if _storage_instance is None or (db_path and _storage_instance.db_path != db_path):
        _storage_instance = SQLiteWorkspaceStorage(db_path or DEFAULT_DB_PATH)
    return _storage_instance


# ── Workspace Manager ─────────────────────────────────────────────────────────

class WorkspaceManager:
    """
    Coordinates workspaces across tasks and sessions.
    Enforces session isolation, manages SQLite persistence, and caches
    in-flight tasks in memory.
    """
    def __init__(self, storage: Optional[WorkspaceStorage] = None):
        self.storage = storage or get_workspace_storage()
        self._workspaces: Dict[str, AgentWorkspace] = {}          # task_id -> workspace
        self._session_latest: Dict[str, str] = {}                 # session_id -> latest task_id

    def create(self, session_id: str, query: str, plan: Optional[dict] = None) -> AgentWorkspace:
        """Create and register a new workspace task."""
        return self.create_task(session_id=session_id, query=query, plan=plan)

    def create_task(
        self,
        session_id: str,
        query: str,
        task_id: Optional[str] = None,
        plan: Optional[dict] = None
    ) -> AgentWorkspace:
        sid = session_id or f"sess_{str(uuid.uuid4())[:8]}"
        tid = task_id or str(uuid.uuid4())[:8]

        ws = AgentWorkspace(
            session_id=sid,
            task_id=tid,
            original_query=query,
            status="pending",
            plan=plan or {},
            _storage=self.storage
        )
        ws.save()

        self._workspaces[tid] = ws
        self._session_latest[sid] = tid
        logger.info(f"🗂️ Task workspace created | task={tid} | session={sid}")
        return ws

    def get(self, session_id: str) -> Optional[AgentWorkspace]:
        """Backward-compatible lookup: gets latest workspace for a session."""
        return self.get_latest_for_session(session_id)

    def get_latest_for_session(self, session_id: str) -> Optional[AgentWorkspace]:
        if not session_id:
            return None
        # Check in-memory mapping first
        if session_id in self._session_latest:
            tid = self._session_latest[session_id]
            if tid in self._workspaces:
                return self._workspaces[tid]

        # Check database
        tasks = self.storage.get_by_session(session_id, limit=1)
        if tasks:
            ws = tasks[0]
            ws._storage = self.storage
            self._workspaces[ws.task_id] = ws
            self._session_latest[session_id] = ws.task_id
            return ws
        return None

    def get_workspace(
        self,
        task_id: str,
        session_id: Optional[str] = None
    ) -> Optional[AgentWorkspace]:
        """
        Retrieve workspace by task_id.
        If session_id is provided, enforces session isolation.
        """
        if not task_id:
            return None

        # Check in-memory cache
        ws = self._workspaces.get(task_id)
        if ws is None:
            # Fallback to database
            ws = self.storage.get(task_id)
            if ws is not None:
                ws._storage = self.storage
                self._workspaces[task_id] = ws

        if ws is None:
            return None

        # Prevent cross-session data leaks
        if session_id and ws.session_id != session_id:
            logger.warning(
                f"🚨 Cross-session access denied: task '{task_id}' belongs to session "
                f"'{ws.session_id}', attempted access by '{session_id}'"
            )
            raise PermissionError(
                f"Access denied: task '{task_id}' does not belong to session '{session_id}'."
            )

        return ws

    def list_session_tasks(self, session_id: str, limit: int = 20) -> List[AgentWorkspace]:
        if not session_id:
            return []
        tasks = self.storage.get_by_session(session_id, limit=limit)
        for t in tasks:
            t._storage = self.storage
        return tasks

    def get_status(self, session_id: str) -> dict:
        ws = self.get_latest_for_session(session_id)
        if not ws:
            return {"status": "idle", "message": "No active task"}
        return ws.summary()

    def clear(self, session_id: str):
        self.clear_session(session_id)

    def clear_session(self, session_id: str):
        """Clear all tasks for a session from memory and storage."""
        if not session_id:
            return
        self.storage.clear_session(session_id)
        # Clean up memory
        to_del = [tid for tid, ws in self._workspaces.items() if ws.session_id == session_id]
        for tid in to_del:
            del self._workspaces[tid]
        if session_id in self._session_latest:
            del self._session_latest[session_id]
        logger.info(f"🗂️ Workspace cleared for session {session_id}")

    def all_sessions(self) -> List[dict]:
        tasks = self.storage.all_tasks(limit=100)
        return [
            {"session_id": t.session_id, "status": t.status, "task_id": t.task_id}
            for t in tasks
        ]


# ── Global Workspace Manager Singleton & Helpers ──────────────────────────────

_workspace_manager_instance: Optional[WorkspaceManager] = None

def get_workspace_manager() -> WorkspaceManager:
    global _workspace_manager_instance
    if _workspace_manager_instance is None:
        _workspace_manager_instance = WorkspaceManager()
    return _workspace_manager_instance


def get_or_create_workspace(
    session_id: Optional[str] = None,
    query: str = "",
    task_id: Optional[str] = None
) -> AgentWorkspace:
    """
    Canonical helper function ensuring workspace is never None.
    Maintains compatibility with all existing callers.
    """
    mgr = get_workspace_manager()
    sid = session_id or f"sess_{str(uuid.uuid4())[:8]}"

    if task_id:
        existing = mgr.get_workspace(task_id, session_id=sid)
        if existing:
            if query and not existing.original_query:
                existing.original_query = query
                existing.save()
            return existing

    # Look for active latest workspace for this session if it hasn't completed and matches query
    latest = mgr.get_latest_for_session(sid)
    if latest and latest.status in ["pending", "running", "planning", "idle"] and not latest.completed_at:
        if not latest.original_query or (query and latest.original_query == query):
            if query and not latest.original_query:
                latest.original_query = query
                latest.save()
            return latest

    # Create new task workspace
    return mgr.create_task(session_id=sid, query=query, task_id=task_id)
