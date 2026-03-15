import os
import uuid
import bcrypt
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_client: Optional[Client] = None


def get_db() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in .env"
            )
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── Chat Sessions ─────────────────────────────────────────────

def create_session(user: str, title: str = "New Chat") -> str:
    """Create a new chat session and return its id."""
    session_id = uuid.uuid4().hex
    get_db().table("chat_sessions").insert({
        "id": session_id,
        "user_id": user,
        "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return session_id


def get_sessions(limit: int = 50):
    """Return all chat sessions, newest-first (shared across all users)."""
    resp = (
        get_db()
        .table("chat_sessions")
        .select("id, title, user_id, created_at, updated_at")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data


def update_session_title(session_id: str, title: str):
    """Update a session's title (auto-set from first user message)."""
    get_db().table("chat_sessions").update({
        "title": title,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", session_id).execute()


def touch_session(session_id: str):
    """Bump updated_at so the session floats to the top."""
    get_db().table("chat_sessions").update({
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", session_id).execute()


def delete_session(session_id: str):
    """Delete a session and all its messages (admin only)."""
    get_db().table("chat_history").delete().eq("session_id", session_id).execute()
    get_db().table("chat_sessions").delete().eq("id", session_id).execute()


# ── Chat History ──────────────────────────────────────────────

def save_message(user: str, role: str, content: str, session_id: str, folder: str | None = None):
    """Save a single chat message."""
    get_db().table("chat_history").insert({
        "user_id": user,
        "role": role,
        "content": content,
        "session_id": session_id,
        "folder": folder,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def get_chat_history(session_id: str, limit: int = 100):
    """Return all messages for a session, oldest-first."""
    resp = (
        get_db()
        .table("chat_history")
        .select("role, content, user_id, folder, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(resp.data))


def clear_chat_history():
    """Delete all chat messages and sessions (admin only)."""
    get_db().table("chat_history").delete().neq("id", "").execute()
    get_db().table("chat_sessions").delete().neq("id", "").execute()


# ── File Metadata ─────────────────────────────────────────────

def save_file_record(user: str, folder: str, filename: str, size_bytes: int):
    """Record an uploaded file in the database."""
    get_db().table("uploaded_files").insert({
        "user_id": user,
        "folder": folder,
        "filename": filename,
        "size_bytes": size_bytes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def get_file_records(user: str, folder: str):
    """List file records for a user+folder."""
    resp = (
        get_db()
        .table("uploaded_files")
        .select("filename, size_bytes, created_at")
        .eq("user_id", user)
        .eq("folder", folder)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data


def delete_file_record(folder: str, filename: str):
    """Remove a file record from the database."""
    (
        get_db()
        .table("uploaded_files")
        .delete()
        .eq("folder", folder)
        .eq("filename", filename)
        .execute()
    )


# ── Users ─────────────────────────────────────────────────────

def create_user(username: str, password: str | None = None):
    """Insert a new user. Pass a password for local auth; omit it for Google OAuth users."""
    if password:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    else:
        hashed = "GOOGLE_OAUTH"  # sentinel — this user logs in via Google only
    get_db().table("users").insert({
        "username": username,
        "password_hash": hashed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def verify_user(username: str, password: str) -> bool:
    """Return True if username exists and password matches."""
    resp = (
        get_db()
        .table("users")
        .select("password_hash")
        .eq("username", username)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return False
    stored_hash = resp.data[0]["password_hash"]
    return bcrypt.checkpw(password.encode(), stored_hash.encode())


def user_exists(username: str) -> bool:
    """Return True if a local user with this username is registered."""
    resp = (
        get_db()
        .table("users")
        .select("username")
        .eq("username", username)
        .limit(1)
        .execute()
    )
    return bool(resp.data)


def list_users():
    """Return all users with username, auth type, and created_at."""
    resp = (
        get_db()
        .table("users")
        .select("username, password_hash, created_at")
        .order("created_at", desc=False)
        .execute()
    )
    return [
        {
            "username": row["username"],
            "auth_type": "google" if row["password_hash"] == "GOOGLE_OAUTH" else "local",
            "created_at": row["created_at"],
        }
        for row in resp.data
    ]


def delete_user(username: str):
    """Remove a local user."""
    get_db().table("users").delete().eq("username", username).execute()
