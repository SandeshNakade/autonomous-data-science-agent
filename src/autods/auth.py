"""Local accounts and opaque cookie sessions for the academic web workspace.

Account provisioning is deliberately a local administrator command, not public
registration. The browser never chooses its own role. No additional auth service
or password dependency is needed for this single-machine prototype.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

COOKIE = "autods_session"
SESSION_SECONDS = 8 * 60 * 60
PASSWORD_ITERATIONS = 600_000
ROLE_LABELS = {
    "scientist": "Data Scientist / ML Engineer",
    "reviewer": "Model Approver / Reviewer",
    "admin": "MLOps Engineer / Administrator",
}
PERMISSIONS = {
    "scientist": {"workspace:read", "datasets:write", "runs:write", "predictions:read"},
    "reviewer": {"workspace:read", "predictions:read", "reviews:write"},
    "admin": {
        "workspace:read", "datasets:write", "runs:write", "predictions:read",
        "monitoring:write", "retraining:write", "reviews:write", "deployments:write",
        "legacy-promotion:write", "users:write",
    },
}


def database_path() -> Path:
    root = Path(os.environ.get("AUTODS_HOME", "."))
    artifacts = Path(os.environ.get("AUTODS_ARTIFACT_DIR", root / "artifacts"))
    return Path(os.environ.get("AUTODS_AUTH_DB", artifacts / "workspace.sqlite3"))


@contextmanager
def database():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.row_factory = sqlite3.Row
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL, role TEXT NOT NULL, password_hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS login_limits (
                identity TEXT PRIMARY KEY, failures INTEGER NOT NULL, retry_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workspace_audit_events (
                id TEXT PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL,
                action TEXT NOT NULL, resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL, reason TEXT
            );
            CREATE INDEX IF NOT EXISTS workspace_audit_events_at
                ON workspace_audit_events(at DESC);
        """)
        yield connection
    finally:
        connection.close()


def _password_hash(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PASSWORD_ITERATIONS)
    return f"{salt}:{digest.hex()}"


def _public_user(row) -> dict:
    return {
        "id": row["id"], "email": row["email"], "name": row["name"],
        "role": row["role"], "role_label": ROLE_LABELS[row["role"]],
        "permissions": sorted(PERMISSIONS[row["role"]]),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_workspace_audit(
    connection: sqlite3.Connection,
    *,
    actor: str,
    action: str,
    resource_id: str,
    reason: str | None = None,
) -> None:
    """Persist a small, password-free workspace-administration audit event."""

    connection.execute(
        "INSERT INTO workspace_audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            secrets.token_hex(16),
            _utc_now(),
            actor,
            action,
            "workspace_user",
            resource_id,
            reason,
        ),
    )


def workspace_settings() -> dict:
    """Return the fixed local-workspace controls without exposing secrets."""

    return {
        "workspace_mode": "local_academic_prototype",
        "authentication": "local_account_sessions",
        "session_hours": SESSION_SECONDS // 3600,
        "user_provisioning": "local_cli_only",
        "password_management": "local_cli_only",
        "roles": [
            {
                "role": role,
                "label": ROLE_LABELS[role],
                "permissions": sorted(PERMISSIONS[role]),
            }
            for role in ROLE_LABELS
        ],
        "guardrails": [
            "Roles are enforced by the API on every protected request.",
            "A web administrator can change another user's role or active state, never passwords.",
            "At least one active administrator must remain in the workspace.",
            "A changed user's current sessions are revoked immediately.",
        ],
    }


def list_workspace_users() -> list[dict]:
    """Return safe account metadata; password hashes and session tokens stay private."""

    with database() as connection:
        rows = connection.execute(
            "SELECT id, email, name, role, active FROM users "
            "ORDER BY active DESC, name COLLATE NOCASE, email COLLATE NOCASE"
        ).fetchall()
    return [{**_public_user(row), "active": bool(row["active"])} for row in rows]


def list_workspace_audit_events(limit: int = 250) -> list[dict]:
    """Expose safely-renderable workspace administration events to the audit view."""

    with database() as connection:
        rows = connection.execute(
            "SELECT id, at, actor, action, resource_type, resource_id, reason "
            "FROM workspace_audit_events ORDER BY at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def update_workspace_user(
    *,
    actor: dict,
    user_id: str,
    role: str | None,
    active: bool | None,
) -> dict:
    """Change another account's role or active state under local-admin safeguards."""

    if not re.fullmatch(r"[a-f0-9]{32}", user_id):
        raise ValueError("Invalid workspace user identifier.")
    if role is None and active is None:
        raise ValueError("Choose a role or account-status change.")
    if role is not None and role not in ROLE_LABELS:
        raise ValueError("Choose a supported workspace role.")
    if user_id == actor["id"]:
        raise ValueError("Administrators cannot change their own role or active state here.")

    with database() as connection:
        target = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if target is None:
            raise ValueError("Workspace user was not found.")
        next_role = role if role is not None else target["role"]
        next_active = bool(target["active"]) if active is None else active
        changing_active_admin = (
            target["role"] == "admin"
            and bool(target["active"])
            and (next_role != "admin" or not next_active)
        )
        if changing_active_admin:
            active_admins = connection.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin' AND active = 1"
            ).fetchone()[0]
            if active_admins <= 1:
                raise ValueError("Keep at least one active administrator in the workspace.")

        changes: list[str] = []
        if next_role != target["role"]:
            changes.append(f"Role changed from {target['role']} to {next_role}.")
        if next_active != bool(target["active"]):
            changes.append("Account activated." if next_active else "Account deactivated.")
        if not changes:
            return {"user": {**_public_user(target), "active": bool(target["active"])}, "changed": False}

        connection.execute(
            "UPDATE users SET role = ?, active = ? WHERE id = ?",
            (next_role, int(next_active), user_id),
        )
        # A role/state change takes effect only after a fresh sign-in.
        connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        _record_workspace_audit(
            connection,
            actor=actor["email"],
            action="workspace_user_updated",
            resource_id=user_id,
            reason=" ".join(changes),
        )
        updated = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        connection.commit()
    return {"user": {**_public_user(updated), "active": bool(updated["active"])}, "changed": True}


def create_user(email: str, name: str, role: str, password: str) -> dict:
    email, name = email.strip().lower(), name.strip()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or len(email) > 254:
        raise ValueError("Enter a valid email address.")
    if not name or len(name) > 100 or role not in ROLE_LABELS:
        raise ValueError("A name and a supported role are required.")
    if not 12 <= len(password) <= 128:
        raise ValueError("Use a password with 12 to 128 characters.")
    user_id = secrets.token_hex(16)
    password_hash = _password_hash(password)
    with database() as connection:
        try:
            connection.execute(
                "INSERT INTO users (id, email, name, role, password_hash) VALUES (?, ?, ?, ?, ?)",
                (user_id, email, name, role, password_hash),
            )
            _record_workspace_audit(
                connection,
                actor="local_cli",
                action="workspace_user_created",
                resource_id=user_id,
                reason=f"Local CLI account created with the {role} role.",
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("An account with that email already exists.") from exc
    return _public_user({"id": user_id, "email": email, "name": name, "role": role})


def require_client_header(request: Request) -> None:
    # Cross-site forms cannot add this header; cross-origin fetch would require
    # a preflight. We intentionally do not enable CORS for these cookie APIs.
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        if request.headers.get("X-AutoDS-Client") != "web":
            raise HTTPException(403, "The workspace request header is required.")
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Cross-origin workspace requests are not allowed.")


def current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE, "")
    if not token or len(token) > 256:
        raise HTTPException(401, "Sign in to your workspace to continue.")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with database() as connection:
        row = connection.execute(
            "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id "
            "WHERE token_hash = ? AND expires_at > ? AND active = 1",
            (token_hash, time.time()),
        ).fetchone()
    if row is None:
        raise HTTPException(401, "Your session has expired. Please sign in again.")
    require_client_header(request)
    return _public_user(row)


def require_permission(permission: str):
    def check(user: dict = Depends(current_user)) -> dict:
        if permission not in user["permissions"]:
            raise HTTPException(403, "Your workspace role does not allow this action.")
        return user
    return check


router = APIRouter(prefix="/auth", tags=["Workspace authentication"])


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class WorkspaceUserUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str | None = Field(default=None, max_length=32)
    active: bool | None = None


@router.get("/status")
def auth_status() -> dict:
    with database() as connection:
        configured = connection.execute("SELECT 1 FROM users WHERE active = 1 LIMIT 1").fetchone()
    return {"configured": bool(configured)}


@router.post("/login", dependencies=[Depends(require_client_header)])
def login(payload: LoginRequest, response: Response) -> dict:
    email = payload.email.strip().lower()
    identity = hashlib.sha256(email.encode()).hexdigest()
    now = time.time()
    with database() as connection:
        limit = connection.execute("SELECT * FROM login_limits WHERE identity = ?", (identity,)).fetchone()
        if limit and limit["failures"] >= 5 and limit["retry_at"] > now:
            raise HTTPException(429, "Too many sign-in attempts. Try again in 15 minutes.")
        row = connection.execute("SELECT * FROM users WHERE email = ? AND active = 1", (email,)).fetchone()
        # Use the same expensive derivation even for a nonexistent account.
        stored = row["password_hash"] if row else "0" * 32 + ":" + "0" * 64
        valid = hmac.compare_digest(_password_hash(payload.password, stored.split(":")[0]), stored)
        if not row or not valid:
            failures = limit["failures"] + 1 if limit and limit["retry_at"] > now else 1
            connection.execute(
                "INSERT OR REPLACE INTO login_limits VALUES (?, ?, ?)",
                (identity, failures, now + 900),
            )
            connection.commit()
            raise HTTPException(401, "Email or password is incorrect.")
        connection.execute("DELETE FROM login_limits WHERE identity = ? OR retry_at <= ?", (identity, now))
        token = secrets.token_urlsafe(32)
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (hashlib.sha256(token.encode()).hexdigest(), row["id"], now + SESSION_SECONDS),
        )
        connection.commit()
    response.set_cookie(
        COOKIE, token, max_age=SESSION_SECONDS, httponly=True, samesite="strict",
        secure=os.environ.get("AUTODS_COOKIE_SECURE", "false").lower() == "true", path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return _public_user(row)


@router.get("/me")
def me(user: dict = Depends(current_user)) -> dict:
    return user


@router.get("/settings")
def settings(user: dict = Depends(current_user)) -> dict:
    """All signed-in users may read the fixed local workspace configuration."""

    return workspace_settings()


@router.get("/users")
def users(user: dict = Depends(require_permission("users:write"))) -> dict:
    """Only administrators may enumerate local workspace accounts."""

    return {"items": list_workspace_users()}


@router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    payload: WorkspaceUserUpdateRequest,
    user: dict = Depends(require_permission("users:write")),
) -> dict:
    """Only administrators may change another account's role or active state."""

    try:
        return update_workspace_user(
            actor=user,
            user_id=user_id,
            role=payload.role,
            active=payload.active,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/logout", dependencies=[Depends(require_client_header)])
def logout(request: Request, response: Response) -> dict:
    token = request.cookies.get(COOKIE, "")
    with database() as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),))
        connection.commit()
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict")
    return {"message": "Signed out successfully."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision a local AutoDS workspace account.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", choices=ROLE_LABELS, required=True)
    args = parser.parse_args()
    password = getpass.getpass("Password (12+ characters): ")
    if password != getpass.getpass("Confirm password: "):
        parser.error("Passwords do not match.")
    try:
        user = create_user(args.email, args.name, args.role, password)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Account created: {user['email']} | {user['role_label']}")
    print("Sign in through the new web workspace. No password is stored in plain text.")


if __name__ == "__main__":
    main()
