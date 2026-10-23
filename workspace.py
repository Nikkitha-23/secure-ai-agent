"""
workspace.py — Agent Workspace / Task State Re-export
------------------------------------------------------
Exposes unified Workspace & Task State management from core.workspace.
Ensures durable SQLite task tracking, session isolation, and
compatibility across all agent loop components.
"""

from core.workspace import (
    WorkspaceStep,
    AgentWorkspace,
    WorkspaceStorage,
    SQLiteWorkspaceStorage,
    WorkspaceManager,
    get_workspace_manager,
    get_or_create_workspace,
    get_workspace_storage,
    truncate_output,
    WORKSPACE_DIR,
    DEFAULT_DB_PATH
)

__all__ = [
    "WorkspaceStep",
    "AgentWorkspace",
    "WorkspaceStorage",
    "SQLiteWorkspaceStorage",
    "WorkspaceManager",
    "get_workspace_manager",
    "get_or_create_workspace",
    "get_workspace_storage",
    "truncate_output",
    "WORKSPACE_DIR",
    "DEFAULT_DB_PATH"
]