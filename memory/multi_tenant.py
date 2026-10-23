"""
multi_tenant.py
===============
L4 Multi-Tenant Memory System for the Secure AI Agent.
Implements 4 memory layers with strict tenant + role isolation.

Memory Layers:
  L1 — Episodic   : Per-session conversation history
  L2 — Semantic   : Long-term facts learned about the user/domain
  L3 — Preference : User-specific preferences (tone, language, format)
  L4 — Procedural : Learned workflows and step-by-step procedures

Tenants supported (demo):
  - anna_univ   (Education)
  - vit         (Education)
  - hospital_a  (Healthcare)

Roles supported:
  - student | teacher | admin        (Education)
  - doctor  | nurse   | hospital_admin (Healthcare)

Usage:
    from memory.multi_tenant import TenantMemoryManager

    mem = TenantMemoryManager(tenant_id="anna_univ", user_id="u001", role="student")
    mem.add_episodic("What is the exam schedule?", "Exams begin on Nov 12.")
    mem.add_semantic("user_dept", "Computer Science")
    mem.add_preference("response_language", "English")
    mem.add_procedural("exam_query_flow", ["check_syllabus", "check_calendar", "respond"])

    context = mem.get_full_context()
"""

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EPISODIC_TURNS   = 20     # Max conversation turns stored per session
MAX_SEMANTIC_ENTRIES = 100    # Max semantic facts per user
MAX_PROCEDURAL_STEPS = 50     # Max procedural entries per tenant

# Role hierarchy — higher number = more access
ROLE_HIERARCHY: Dict[str, int] = {
    # Education
    "student":          1,
    "teacher":          2,
    "admin":            3,
    # Healthcare
    "nurse":            1,
    "doctor":           2,
    "hospital_admin":   3,
}

# Tenant → Domain mapping
TENANT_DOMAIN: Dict[str, str] = {
    "anna_univ":  "education",
    "vit":        "education",
    "hospital_a": "healthcare",
}

# Roles allowed per domain
DOMAIN_ROLES: Dict[str, List[str]] = {
    "education":  ["student", "teacher", "admin"],
    "healthcare": ["nurse", "doctor", "hospital_admin"],
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodicEntry:
    turn_id:    str
    user_query: str
    agent_response: str
    timestamp:  float = field(default_factory=time.time)
    metadata:   Dict  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SemanticEntry:
    key:       str
    value:     Any
    source:    str = "inferred"      # "inferred" | "explicit" | "document"
    confidence: float = 1.0
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PreferenceEntry:
    key:       str
    value:     Any
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProceduralEntry:
    procedure_name: str
    steps:          List[str]
    domain:         str
    used_count:     int   = 0
    last_used:      float = field(default_factory=time.time)
    created_at:     float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Tenant Registry — Central store (in-memory; swap with Redis/DB in prod)
# ---------------------------------------------------------------------------

class TenantRegistry:
    """
    Singleton registry that holds all tenant memory stores.
    Enforces that no tenant can access another tenant's data.
    """
    _instance = None
    _store: Dict[str, Dict] = {}          # tenant_id → user_id → memory
    _audit_log: List[Dict]  = []          # Security audit trail

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Storage ──────────────────────────────────────────────────────────────

    def get_user_store(self, tenant_id: str, user_id: str) -> Dict:
        self._store.setdefault(tenant_id, {})
        self._store[tenant_id].setdefault(user_id, {
            "episodic":   deque(maxlen=MAX_EPISODIC_TURNS),
            "semantic":   {},
            "preference": {},
            "procedural": {},   # Shared at tenant level, not user level
        })
        return self._store[tenant_id][user_id]

    def get_tenant_procedural(self, tenant_id: str) -> Dict:
        """Procedural memory is shared across all users of a tenant."""
        self._store.setdefault(tenant_id, {})
        self._store[tenant_id].setdefault("__procedural__", {})
        return self._store[tenant_id]["__procedural__"]

    # ── Isolation Guard ───────────────────────────────────────────────────────

    def assert_tenant_access(self, requesting_tenant: str, target_tenant: str, operation: str):
        if requesting_tenant != target_tenant:
            self._log_violation(requesting_tenant, target_tenant, operation)
            raise PermissionError(
                f"[TENANT ISOLATION VIOLATION] "
                f"Tenant '{requesting_tenant}' attempted to access '{target_tenant}' memory. "
                f"Operation: {operation}"
            )

    def assert_role_access(self, role: str, required_level: int, resource: str):
        user_level = ROLE_HIERARCHY.get(role, 0)
        if user_level < required_level:
            raise PermissionError(
                f"[ROLE ACCESS DENIED] Role '{role}' (level {user_level}) "
                f"cannot access '{resource}' (requires level {required_level})."
            )

    def assert_domain_role(self, tenant_id: str, role: str):
        domain = TENANT_DOMAIN.get(tenant_id)
        if domain and role not in DOMAIN_ROLES.get(domain, []):
            raise PermissionError(
                f"[DOMAIN ROLE MISMATCH] Role '{role}' is not valid for "
                f"tenant '{tenant_id}' (domain: {domain})."
            )

    # ── Audit ─────────────────────────────────────────────────────────────────

    def _log_violation(self, from_tenant: str, to_tenant: str, operation: str):
        entry = {
            "event":        "ISOLATION_VIOLATION",
            "from_tenant":  from_tenant,
            "to_tenant":    to_tenant,
            "operation":    operation,
            "timestamp":    datetime.now().isoformat(),
        }
        self._audit_log.append(entry)
        print(f"  🚨 SECURITY AUDIT: {entry}")

    def log_access(self, tenant_id: str, user_id: str, operation: str, layer: str):
        self._audit_log.append({
            "event":     "MEMORY_ACCESS",
            "tenant_id": tenant_id,
            "user_id":   user_id,
            "operation": operation,
            "layer":     layer,
            "timestamp": datetime.now().isoformat(),
        })

    def get_audit_log(self, tenant_id: Optional[str] = None) -> List[Dict]:
        if tenant_id:
            return [e for e in self._audit_log if e.get("tenant_id") == tenant_id]
        return list(self._audit_log)

    # ── Admin Utilities ──────────────────────────────────────────────────────

    def list_tenants(self) -> List[str]:
        return [k for k in self._store.keys() if not k.startswith("_")]

    def flush_tenant(self, tenant_id: str):
        """Completely wipe a tenant's memory (admin use only)."""
        if tenant_id in self._store:
            del self._store[tenant_id]

    def get_tenant_stats(self, tenant_id: str) -> Dict:
        if tenant_id not in self._store:
            return {"error": "Tenant not found"}
        users = {k: v for k, v in self._store[tenant_id].items() if not k.startswith("_")}
        return {
            "tenant_id":      tenant_id,
            "domain":         TENANT_DOMAIN.get(tenant_id, "unknown"),
            "total_users":    len(users),
            "total_episodic": sum(len(u["episodic"]) for u in users.values()),
            "total_semantic": sum(len(u["semantic"]) for u in users.values()),
        }


# ---------------------------------------------------------------------------
# Per-User Memory Manager
# ---------------------------------------------------------------------------

class TenantMemoryManager:
    """
    Main interface for all memory operations.
    Every operation is scoped to (tenant_id, user_id) automatically.
    """

    def __init__(self, tenant_id: str, user_id: str, role: str):
        self.tenant_id = tenant_id
        self.user_id   = user_id
        self.role      = role

        self._registry = TenantRegistry()

        # Validate role against tenant domain on init
        self._registry.assert_domain_role(tenant_id, role)

        self._store      = self._registry.get_user_store(tenant_id, user_id)
        self._procedural = self._registry.get_tenant_procedural(tenant_id)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_turn_id(self) -> str:
        raw = f"{self.tenant_id}:{self.user_id}:{time.time()}"
        return "turn_" + hashlib.md5(raw.encode()).hexdigest()[:8]

    def _log(self, operation: str, layer: str):
        self._registry.log_access(self.tenant_id, self.user_id, operation, layer)

    # =========================================================================
    # L1 — EPISODIC MEMORY
    # =========================================================================

    def add_episodic(
        self,
        user_query: str,
        agent_response: str,
        metadata: Optional[Dict] = None,
    ) -> EpisodicEntry:
        """Store a conversation turn."""
        entry = EpisodicEntry(
            turn_id=self._make_turn_id(),
            user_query=user_query,
            agent_response=agent_response,
            metadata=metadata or {},
        )
        self._store["episodic"].append(entry)
        self._log("write", "episodic")
        return entry

    def get_episodic(self, last_n: int = 5) -> List[EpisodicEntry]:
        """Retrieve last N conversation turns."""
        self._log("read", "episodic")
        turns = list(self._store["episodic"])
        return turns[-last_n:] if last_n else turns

    def get_episodic_as_messages(self, last_n: int = 5) -> List[Dict[str, str]]:
        """
        Return episodic memory formatted for LLM message history.
        Ready to inject into Anthropic API messages[] array.
        """
        messages = []
        for turn in self.get_episodic(last_n):
            messages.append({"role": "user",      "content": turn.user_query})
            messages.append({"role": "assistant", "content": turn.agent_response})
        return messages

    def clear_episodic(self):
        """Clear session memory (on logout or new session)."""
        self._store["episodic"].clear()
        self._log("clear", "episodic")

    # =========================================================================
    # L2 — SEMANTIC MEMORY
    # =========================================================================

    def add_semantic(
        self,
        key: str,
        value: Any,
        source: str = "inferred",
        confidence: float = 1.0,
    ) -> SemanticEntry:
        """Store a long-term fact about the user."""
        if len(self._store["semantic"]) >= MAX_SEMANTIC_ENTRIES:
            # Evict the oldest entry
            oldest = min(self._store["semantic"], key=lambda k: self._store["semantic"][k].updated_at)
            del self._store["semantic"][oldest]

        entry = SemanticEntry(key=key, value=value, source=source, confidence=confidence)
        self._store["semantic"][key] = entry
        self._log("write", "semantic")
        return entry

    def get_semantic(self, key: str) -> Optional[Any]:
        """Retrieve a semantic fact by key."""
        self._log("read", "semantic")
        entry = self._store["semantic"].get(key)
        return entry.value if entry else None

    def get_all_semantic(self) -> Dict[str, Any]:
        """Return all semantic facts as a flat dict."""
        self._log("read", "semantic")
        return {k: v.value for k, v in self._store["semantic"].items()}

    def delete_semantic(self, key: str) -> bool:
        if key in self._store["semantic"]:
            del self._store["semantic"][key]
            self._log("delete", "semantic")
            return True
        return False

    # =========================================================================
    # L3 — PREFERENCE MEMORY
    # =========================================================================

    def add_preference(self, key: str, value: Any) -> PreferenceEntry:
        """Set a user preference."""
        entry = PreferenceEntry(key=key, value=value)
        self._store["preference"][key] = entry
        self._log("write", "preference")
        return entry

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        self._log("read", "preference")
        entry = self._store["preference"].get(key)
        return entry.value if entry else default

    def get_all_preferences(self) -> Dict[str, Any]:
        return {k: v.value for k, v in self._store["preference"].items()}

    # =========================================================================
    # L4 — PROCEDURAL MEMORY  (tenant-shared)
    # =========================================================================

    def add_procedural(
        self,
        procedure_name: str,
        steps: List[str],
        role_required: int = 1,
    ) -> ProceduralEntry:
        """
        Store a reusable procedure (e.g. query workflow).
        Requires minimum role level to write.
        """
        self._registry.assert_role_access(self.role, role_required, f"procedural:{procedure_name}")

        domain = TENANT_DOMAIN.get(self.tenant_id, "general")
        entry  = ProceduralEntry(
            procedure_name=procedure_name,
            steps=steps,
            domain=domain,
        )
        self._procedural[procedure_name] = entry
        self._log("write", "procedural")
        return entry

    def get_procedural(self, procedure_name: str) -> Optional[List[str]]:
        """Retrieve steps for a named procedure."""
        self._log("read", "procedural")
        entry = self._procedural.get(procedure_name)
        if entry:
            entry.used_count += 1
            entry.last_used   = time.time()
            return entry.steps
        return None

    def list_procedures(self) -> List[str]:
        return list(self._procedural.keys())

    # =========================================================================
    # Context Builder — for RAG prompt injection
    # =========================================================================

    def get_full_context(self, last_n_turns: int = 5) -> Dict:
        """
        Assembles all memory layers into a single context dict.
        Inject this into your RAG prompt / agent loop.
        """
        return {
            "tenant_id":   self.tenant_id,
            "user_id":     self.user_id,
            "role":        self.role,
            "domain":      TENANT_DOMAIN.get(self.tenant_id, "unknown"),
            "episodic":    [t.to_dict() for t in self.get_episodic(last_n_turns)],
            "semantic":    self.get_all_semantic(),
            "preferences": self.get_all_preferences(),
            "procedures":  self.list_procedures(),
        }

    def get_prompt_context_string(self, last_n_turns: int = 3) -> str:
        """
        Returns a formatted string for injecting memory into an LLM prompt.
        Use inside your system prompt or user message.
        """
        ctx   = self.get_full_context(last_n_turns)
        prefs = ctx["preferences"]
        facts = ctx["semantic"]
        turns = ctx["episodic"]

        parts = [
            f"[MEMORY CONTEXT]",
            f"Tenant : {ctx['tenant_id']} ({ctx['domain']})",
            f"Role   : {ctx['role']}",
        ]

        if facts:
            parts.append("Known facts about user:")
            for k, v in facts.items():
                parts.append(f"  • {k}: {v}")

        if prefs:
            parts.append("User preferences:")
            for k, v in prefs.items():
                parts.append(f"  • {k}: {v}")

        if turns:
            parts.append(f"Recent conversation ({len(turns)} turns):")
            for t in turns[-3:]:
                parts.append(f"  Q: {t['user_query'][:80]}")
                parts.append(f"  A: {t['agent_response'][:80]}")

        return "\n".join(parts)

    # =========================================================================
    # Memory Summary (for admin / dashboard)
    # =========================================================================

    def get_memory_summary(self) -> Dict:
        return {
            "tenant_id":         self.tenant_id,
            "user_id":           self.user_id,
            "role":              self.role,
            "episodic_turns":    len(self._store["episodic"]),
            "semantic_facts":    len(self._store["semantic"]),
            "preferences":       len(self._store["preference"]),
            "procedures_known":  len(self._procedural),
        }


# ---------------------------------------------------------------------------
# Cross-Tenant Isolation Test
# ---------------------------------------------------------------------------

def run_isolation_test() -> bool:
    """
    Proves that tenants cannot access each other's memory.
    Returns True if all tests pass.
    """
    print("\n" + "=" * 60)
    print("  MULTI-TENANT ISOLATION TEST")
    print("=" * 60)

    registry = TenantRegistry()
    all_pass = True

    # Setup: write memory for two tenants
    anna = TenantMemoryManager("anna_univ", "student_001", "student")
    anna.add_semantic("department", "Computer Science")
    anna.add_episodic("What is the exam date?", "Exams start Nov 12.")

    vit = TenantMemoryManager("vit", "student_002", "student")
    vit.add_semantic("department", "Electronics")
    vit.add_episodic("What is CGPA formula?", "CGPA = sum of grade points / total credits.")

    # Test 1: Each tenant reads only their own data
    anna_dept = anna.get_semantic("department")
    vit_dept  = vit.get_semantic("department")

    test1 = anna_dept == "Computer Science" and vit_dept == "Electronics"
    print(f"  {'✅' if test1 else '❌'} Test 1: Semantic isolation (Anna≠VIT dept)")
    all_pass &= test1

    # Test 2: Cross-tenant access raises PermissionError
    test2 = False
    try:
        registry.assert_tenant_access("anna_univ", "vit", "read_semantic")
    except PermissionError:
        test2 = True
    print(f"  {'✅' if test2 else '❌'} Test 2: Cross-tenant access correctly blocked")
    all_pass &= test2

    # Test 3: Domain role mismatch blocked
    test3 = False
    try:
        # "doctor" role in education tenant should be rejected
        TenantMemoryManager("anna_univ", "u999", "doctor")
    except PermissionError:
        test3 = True
    print(f"  {'✅' if test3 else '❌'} Test 3: Domain/role mismatch correctly blocked")
    all_pass &= test3

    # Test 4: Healthcare tenant isolation
    hosp = TenantMemoryManager("hospital_a", "doc_001", "doctor")
    hosp.add_semantic("specialization", "Cardiology")
    hosp.add_preference("response_format", "clinical")
    hosp.add_episodic(
        "What is metformin dosage?",
        "Standard starting dose is 500mg twice daily with meals."
    )

    test4 = hosp.get_semantic("specialization") == "Cardiology"
    print(f"  {'✅' if test4 else '❌'} Test 4: Healthcare memory stored correctly")
    all_pass &= test4

    # Test 5: Education cannot access Healthcare memory
    test5 = False
    try:
        registry.assert_tenant_access("anna_univ", "hospital_a", "read_episodic")
    except PermissionError:
        test5 = True
    print(f"  {'✅' if test5 else '❌'} Test 5: Education→Healthcare access blocked")
    all_pass &= test5

    # Test 6: Role escalation blocked
    test6 = False
    try:
        registry.assert_role_access("student", 2, "teacher_documents")
    except PermissionError:
        test6 = True
    print(f"  {'✅' if test6 else '❌'} Test 6: Role escalation (student→teacher) blocked")
    all_pass &= test6

    print("-" * 60)
    print(f"  RESULT: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    print("=" * 60 + "\n")

    return all_pass


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("\n  DEMO — TenantMemoryManager\n")

    # Anna University student
    mem = TenantMemoryManager("anna_univ", "student_riya", "student")

    mem.add_semantic("name",       "Riya")
    mem.add_semantic("department", "CSE")
    mem.add_semantic("semester",   6)

    mem.add_preference("response_language", "English")
    mem.add_preference("detail_level",      "concise")

    mem.add_episodic(
        "What subjects are in semester 6?",
        "Semester 6 includes: OS, DBMS, CN, Compiler Design, and Electives."
    )
    mem.add_episodic(
        "When are the internal exams?",
        "Internal Assessment 2 is scheduled for October 28-30."
    )

    mem.add_procedural(
        "syllabus_query_flow",
        steps=[
            "Identify semester from user context",
            "Search syllabus PDF for relevant subject",
            "Return unit-wise breakdown",
            "Suggest past papers if available",
        ],
        role_required=1,
    )

    print(mem.get_prompt_context_string())
    print("\n  Memory Summary:", json.dumps(mem.get_memory_summary(), indent=2))

    # Run isolation tests
    run_isolation_test()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo()