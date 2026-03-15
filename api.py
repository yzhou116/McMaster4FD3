import os
import shutil
import httpx
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from authlib.integrations.httpx_client import AsyncOAuth2Client
import secrets

# Load environment variables
load_dotenv()

# Database
from database import (
    save_message, get_chat_history, clear_chat_history,
    save_file_record, get_file_records, delete_file_record,
    create_session, get_sessions, update_session_title,
    touch_session, delete_session,
    create_user, verify_user, user_exists, list_users, delete_user,
)

# Import your server functions
from server import mcp, STATE, ingest_docs_impl, ask_impl, main_agent_router

mcp_app = mcp.http_app(path="/")
api = FastAPI(lifespan=mcp_app.lifespan)

# --- Middleware (Only once) ---
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
# All users share a single vault; uploads/messages are tagged by user_id for audit
SHARED_VAULT = os.path.join(os.getcwd(), "shared_vault")
os.makedirs(SHARED_VAULT, exist_ok=True)

# Google OAuth2 Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class ChatRequest(BaseModel):    
    message: str
    session_id: str | None = None
    docs_dir: str | None = None

# --- Auth Helpers ---
def get_current_user(token: str = Depends(oauth2_scheme)):
    """All tokens — username or Google email — must be in the users table."""
    if not user_exists(token):
        raise HTTPException(status_code=401, detail="Access denied. Contact your administrator.")
    return token

def require_admin(current_user: str = Depends(get_current_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# --- Endpoints ---

@api.get("/")
async def serve_index():
    # Serves the HTML file directly
    return FileResponse('chatwindow.html')

@api.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not verify_user(form_data.username, form_data.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": form_data.username, "token_type": "bearer"}


# ── User Management Endpoints (authenticated users only) ──────

class UserCreate(BaseModel):
    username: str
    password: str | None = None  # None = Google OAuth user (no password needed)

@api.get("/users")
def get_users(current_user: str = Depends(require_admin)):
    """List all registered local users. Admin only."""
    return list_users()

@api.post("/users", status_code=201)
def add_user(body: UserCreate, current_user: str = Depends(require_admin)):
    """Add a new user. Admin only. Omit password for Google OAuth users."""
    if user_exists(body.username):
        raise HTTPException(status_code=409, detail="Username already exists")
    create_user(body.username, body.password)
    return {"status": "created", "username": body.username}

@api.delete("/users/{username}")
def remove_user(username: str, current_user: str = Depends(require_admin)):
    """Remove a local user. Admin only."""
    if username == "admin":
        raise HTTPException(status_code=400, detail="Cannot remove the admin account")
    if not user_exists(username):
        raise HTTPException(status_code=404, detail="User not found")
    delete_user(username)
    return {"status": "deleted", "username": username}

@api.get("/auth/google")
async def google_login():
    """Redirect to Google OAuth2 consent screen"""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth2 not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")
    
    state = secrets.token_urlsafe(32)
    # Store state in session (in production, use a real session store)
    
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={GOOGLE_REDIRECT_URI}&"
        f"response_type=code&"
        f"scope=openid%20email%20profile&"
        f"state={state}"
    )
    return RedirectResponse(url=google_auth_url)

@api.get("/auth/google/callback")
async def google_callback(code: str, state: str):
    """Handle Google OAuth2 callback"""
    if not code:
        raise HTTPException(status_code=400, detail="No authorization code provided")
    
    # Exchange code for token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": GOOGLE_REDIRECT_URI,
            }
        )
        
        if token_response.status_code != 200:
            error_msg = "Failed to exchange code for token"
            return FileResponse('chatwindow.html', headers={"X-Login-Error": error_msg})
        
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        
        # Get user info from Google
        user_info_response = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if user_info_response.status_code != 200:
            error_msg = "Failed to get user info"
            return FileResponse('chatwindow.html', headers={"X-Login-Error": error_msg})
        
        user_info = user_info_response.json()
        email = user_info.get("email")
        name = user_info.get("name", "User")

        # Block any Google account not explicitly approved by admin
        if not user_exists(email):
            return HTMLResponse("""
            <!DOCTYPE html><html><head><title>Access Denied</title></head>
            <body style="font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;background:#06080f;color:#eef0f6;">
              <div style="text-align:center;">
                <h2 style="color:#ef4444;">Access Denied</h2>
                <p>Your Google account is not authorised. Contact your administrator.</p>
                <a href="/" style="color:#6366f1;">Go back</a>
              </div>
            </body></html>
            """, status_code=403)

        # Return HTML page that sets localStorage and redirects
        html_response = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Google Login Success</title>
        </head>
        <body>
            <script>
                // Store the token and user info in localStorage
                localStorage.setItem('token', '{email}');
                localStorage.setItem('userName', '{name}');
                localStorage.setItem('userEmail', '{email}');
                localStorage.setItem('loginMethod', 'google');
                
                // Redirect to main page
                window.location.href = '/';
            </script>
        </body>
        </html>
        """
        return HTMLResponse(html_response)

@api.get("/folders")
async def list_folders(user: str = Depends(get_current_user)):
    if not os.path.exists(SHARED_VAULT):
        return []
    return [d for d in os.listdir(SHARED_VAULT) if os.path.isdir(os.path.join(SHARED_VAULT, d))]

@api.post("/folders/{folder_name}")
async def create_folder(folder_name: str, user: str = Depends(get_current_user)):
    safe_name = "".join([c for c in folder_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_name:
        raise HTTPException(400, "Invalid folder name")
    folder_path = os.path.join(SHARED_VAULT, safe_name)
    os.makedirs(folder_path, exist_ok=True)
    return {"status": "created"}

@api.post("/upload/{folder_name}")
async def upload_files_endpoint(folder_name: str, files: List[UploadFile] = File(...), user: str = Depends(get_current_user)):
    folder_path = os.path.join(SHARED_VAULT, folder_name)
    if not os.path.exists(folder_path):
        raise HTTPException(404, "Folder not found")

    saved_files = []
    for file in files:
        file_location = os.path.join(folder_path, file.filename)
        contents = await file.read()
        with open(file_location, "wb") as file_object:
            file_object.write(contents)
        saved_files.append(file.filename)
        # Record upload with user_id for audit trail
        save_file_record(user, folder_name, file.filename, len(contents))

    ingest_docs_impl(folder_path, force=True)
    return {"status": "success", "files": saved_files}

@api.get("/files/{folder_name}")
async def list_files(folder_name: str, user: str = Depends(get_current_user)):
    folder_path = os.path.join(SHARED_VAULT, folder_name)
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

@api.delete("/files/{folder_name}/{file_name}")
async def delete_file(folder_name: str, file_name: str, user: str = Depends(get_current_user)):
    folder_path = os.path.join(SHARED_VAULT, folder_name)
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(file_path)
        ingest_docs_impl(folder_path, force=True)
        delete_file_record(folder_name, file_name)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/chat")
def chat(req: ChatRequest, user: str = Depends(get_current_user)):
    # 1. Resolve target dir safely within shared vault — never trust full client path
    if req.docs_dir:
        folder_name = os.path.basename(req.docs_dir.rstrip("/\\"))
        target_dir = os.path.join(SHARED_VAULT, folder_name)
    else:
        target_dir = SHARED_VAULT

    # 2. Ingest
    result = ingest_docs_impl(target_dir, force=False)
    
    if not result.get("ok", False):
        return {"answer": f"Error ingesting docs: {result.get('error')}"}

    # 3. Session handling
    session_id = req.session_id
    is_new_session = False
    if not session_id:
        session_id = create_session(user)
        is_new_session = True

    # 4. Save user message & get answer
    folder = req.docs_dir or "default"
    save_message(user, "user", req.message, session_id, folder)

    # Auto-title the session from the first user message
    if is_new_session:
        title = req.message[:60] + ("..." if len(req.message) > 60 else "")
        update_session_title(session_id, title)

    answer = main_agent_router(req.message)

    save_message(user, "assistant", answer, session_id, folder)
    touch_session(session_id)
    return {"answer": answer, "session_id": session_id}


# ── Session Endpoints ─────────────────────────────────────────

@api.get("/sessions")
def list_sessions(user: str = Depends(get_current_user)):
    """Return all shared chat sessions, newest-first."""
    return get_sessions()


@api.delete("/sessions/{session_id}")
def remove_session(session_id: str, user: str = Depends(require_admin)):
    """Delete a single chat session and its messages. Admin only."""
    delete_session(session_id)
    return {"status": "deleted"}


# ── Chat History Endpoints ────────────────────────────────────

@api.get("/chat/history/{session_id}")
def chat_history(session_id: str, user: str = Depends(get_current_user)):
    """Return messages for a specific session."""
    return get_chat_history(session_id)


@api.delete("/chat/history")
def chat_history_clear(user: str = Depends(require_admin)):
    """Clear all chat history. Admin only."""
    clear_chat_history()
    return {"status": "cleared"}


# Mount MCP (Optional)
api.mount("/mcp", mcp_app)