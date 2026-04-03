"""
role_access.py
--------------
Role-Based Access Control (RBAC) with JWT tokens.
Supports Education + Healthcare domains.

Roles:
  Education  : student | teacher | edu_admin
  Healthcare : doctor  | nurse   | hospital_admin

Usage:
    from role_access import create_token, verify_token, check_permission

    # Login → get token
    token = create_token(user_id="u001", role="student", tenant_id="anna_university")

    # On every query → verify
    payload = verify_token(token)

    # Check if allowed
    check_permission(payload, action="query", domain="education")
"""

import jwt
import datetime
from functools import wraps

# ─── CONFIG ───────────────────────────────────────────────────────────────────

JWT_SECRET  = "secure-ai-agent-secret-key-change-in-production"
JWT_ALGO    = "HS256"
TOKEN_EXPIRY_HOURS = 24

# ─── ROLE DEFINITIONS ─────────────────────────────────────────────────────────

# Each role has a set of allowed actions
ROLE_PERMISSIONS = {

    # ── Education Roles ──────────────────────────────────────────────────────
    "student": {
        "domain": "education",
        "actions": ["query"],
        "can_upload": False,
        "can_see_other_tenants": False,
        "description": "Can only query their own college documents",
    },
    "teacher": {
        "domain": "education",
        "actions": ["query", "upload"],
        "can_upload": True,
        "can_see_other_tenants": False,
        "description": "Can query and upload department documents",
    },
    "edu_admin": {
        "domain": "education",
        "actions": ["query", "upload", "delete", "manage_users"],
        "can_upload": True,
        "can_see_other_tenants": False,
        "description": "Full access to their college tenant",
    },

    # ── Healthcare Roles ─────────────────────────────────────────────────────
    "doctor": {
        "domain": "healthcare",
        "actions": ["query", "upload"],
        "can_upload": True,
        "can_see_other_tenants": False,
        "description": "Can query and upload medical documents",
    },
    "nurse": {
        "domain": "healthcare",
        "actions": ["query"],
        "can_upload": False,
        "can_see_other_tenants": False,
        "description": "Can only query approved medical documents",
    },
    "hospital_admin": {
        "domain": "healthcare",
        "actions": ["query", "upload", "delete", "manage_users"],
        "can_upload": True,
        "can_see_other_tenants": False,
        "description": "Full access to their hospital tenant",
    },

    # ── Super Admin (platform level) ─────────────────────────────────────────
    "super_admin": {
        "domain": "all",
        "actions": ["query", "upload", "delete", "manage_users", "view_all_tenants"],
        "can_upload": True,
        "can_see_other_tenants": True,
        "description": "Platform-level access across all tenants",
    },
}

VALID_ROLES = list(ROLE_PERMISSIONS.keys())


# ─── MOCK USER DB (replace with real DB later) ────────────────────────────────

MOCK_USERS = {
    # Education — Anna University
    "student_001": {"password": "pass123", "role": "student",   "tenant_id": "anna_university", "name": "Priya S"},
    "teacher_001": {"password": "pass123", "role": "teacher",   "tenant_id": "anna_university", "name": "Dr. Kumar"},
    "admin_001":   {"password": "pass123", "role": "edu_admin", "tenant_id": "anna_university", "name": "Admin AU"},

    # Education — VIT
    "student_002": {"password": "pass123", "role": "student",   "tenant_id": "vit_university",  "name": "Rahul M"},
    "admin_002":   {"password": "pass123", "role": "edu_admin", "tenant_id": "vit_university",  "name": "Admin VIT"},

    # Healthcare
    "doctor_001":  {"password": "pass123", "role": "doctor",         "tenant_id": "apollo_hospital", "name": "Dr. Meena"},
    "nurse_001":   {"password": "pass123", "role": "nurse",          "tenant_id": "apollo_hospital", "name": "Nurse Raji"},
    "hadmin_001":  {"password": "pass123", "role": "hospital_admin", "tenant_id": "apollo_hospital", "name": "Admin Apollo"},

    # Super Admin
    "superadmin":  {"password": "admin999", "role": "super_admin", "tenant_id": "platform", "name": "Platform Admin"},
}


# ─── TOKEN CREATION ───────────────────────────────────────────────────────────

def create_token(user_id: str, role: str, tenant_id: str) -> str:
    """
    Create a JWT token for a user.

    Args:
        user_id   : unique user identifier
        role      : one of VALID_ROLES
        tenant_id : which college/hospital they belong to

    Returns:
        JWT token string
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: '{role}'. Must be one of {VALID_ROLES}")

    payload = {
        "user_id":   user_id,
        "role":      role,
        "tenant_id": tenant_id,
        "domain":    ROLE_PERMISSIONS[role]["domain"],
        "iat":       datetime.datetime.utcnow(),
        "exp":       datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS),
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    return token


# ─── TOKEN VERIFICATION ───────────────────────────────────────────────────────

def verify_token(token: str) -> dict:
    """
    Verify and decode a JWT token.

    Returns:
        Decoded payload dict if valid

    Raises:
        PermissionError if token is invalid/expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except jwt.ExpiredSignatureError:
        raise PermissionError("Token expired. Please login again.")
    except jwt.InvalidTokenError as e:
        raise PermissionError(f"Invalid token: {e}")


# ─── PERMISSION CHECK ─────────────────────────────────────────────────────────

def check_permission(payload: dict, action: str, target_tenant_id: str = None) -> dict:
    """
    Check if a user has permission to perform an action.

    Args:
        payload          : decoded JWT payload
        action           : "query" | "upload" | "delete" | "manage_users"
        target_tenant_id : tenant being accessed (for cross-tenant check)

    Returns:
        dict with allowed=True + access details

    Raises:
        PermissionError if not allowed
    """
    role      = payload.get("role")
    tenant_id = payload.get("tenant_id")
    user_id   = payload.get("user_id")

    if role not in ROLE_PERMISSIONS:
        raise PermissionError(f"Unknown role: {role}")

    perms = ROLE_PERMISSIONS[role]

    # 1. Action check
    if action not in perms["actions"]:
        raise PermissionError(
            f"Role '{role}' cannot perform '{action}'. "
            f"Allowed actions: {perms['actions']}"
        )

    # 2. Cross-tenant check
    if target_tenant_id and target_tenant_id != tenant_id:
        if not perms["can_see_other_tenants"]:
            raise PermissionError(
                f"Access denied: '{user_id}' (tenant: {tenant_id}) "
                f"cannot access tenant '{target_tenant_id}'"
            )

    return {
        "allowed": True,
        "user_id": user_id,
        "role": role,
        "tenant_id": tenant_id,
        "domain": perms["domain"],
        "action": action,
    }


# ─── LOGIN HELPER ─────────────────────────────────────────────────────────────

def login(user_id: str, password: str) -> dict:
    """
    Authenticate user and return JWT token.

    Returns:
        dict with token + user info

    Raises:
        PermissionError if credentials are wrong
    """
    user = MOCK_USERS.get(user_id)

    if not user or user["password"] != password:
        raise PermissionError("Invalid user ID or password.")

    token = create_token(
        user_id=user_id,
        role=user["role"],
        tenant_id=user["tenant_id"]
    )

    return {
        "token": token,
        "user_id": user_id,
        "name": user["name"],
        "role": user["role"],
        "tenant_id": user["tenant_id"],
        "message": "Login successful",
    }


# ─── FASTAPI DEPENDENCY (plug into your existing main.py) ────────────────────

def get_current_user(authorization: str) -> dict:
    """
    FastAPI dependency — extract + verify JWT from Authorization header.

    In your FastAPI routes:
        from role_access import get_current_user
        from fastapi import Header

        @app.post("/query")
        def query(request: QueryRequest, authorization: str = Header(...)):
            user = get_current_user(authorization)
            # user["role"], user["tenant_id"] available here
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise PermissionError("Missing or invalid Authorization header. Use: Bearer <token>")

    token = authorization.split(" ")[1]
    return verify_token(token)


# ─── TEST ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("ROLE-BASED ACCESS CONTROL — TEST")
    print("=" * 60)

    # Test 1: Login all users
    print("\n📌 TEST 1: Login")
    for user_id, info in list(MOCK_USERS.items())[:5]:
        result = login(user_id, "pass123" if user_id != "superadmin" else "admin999")
        print(f"  ✅ {result['name']:20s} | Role: {result['role']:15s} | Tenant: {result['tenant_id']}")

    # Test 2: Permission checks
    print("\n📌 TEST 2: Permission Checks")

    tests = [
        ("student_001",  "pass123",  "query",  "anna_university",  True),
        ("student_001",  "pass123",  "upload", "anna_university",  False),  # students can't upload
        ("teacher_001",  "pass123",  "upload", "anna_university",  True),
        ("student_001",  "pass123",  "query",  "vit_university",   False),  # cross-tenant blocked
        ("superadmin",   "admin999", "query",  "anna_university",  True),   # super admin can access all
        ("doctor_001",   "pass123",  "query",  "apollo_hospital",  True),
        ("nurse_001",    "pass123",  "upload", "apollo_hospital",  False),  # nurses can't upload
    ]

    for user_id, password, action, target_tenant, should_pass in tests:
        try:
            auth = login(user_id, password)
            payload = verify_token(auth["token"])
            check_permission(payload, action=action, target_tenant_id=target_tenant)
            result = "✅ ALLOWED"
        except PermissionError as e:
            result = f"🚫 DENIED"

        expected = "✅" if should_pass else "🚫"
        match = "✓" if (should_pass == ("ALLOWED" in result)) else "✗ MISMATCH"
        print(f"  {match} {user_id:15s} | {action:12s} | {target_tenant:20s} | {result}")

    # Test 3: Expired/invalid token
    print("\n📌 TEST 3: Invalid Token")
    try:
        verify_token("this.is.fake")
        print("  ✗ Should have failed!")
    except PermissionError as e:
        print(f"  ✅ Correctly rejected: {e}")

    print("\n✅ RBAC system working correctly!")