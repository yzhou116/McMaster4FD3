import os
import shutil
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List

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
UPLOAD_ROOT = os.path.join(os.getcwd(), "user_vaults")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Simple User DB (In production, use a real DB)
USERS_DB = {
    "admin": "password123",
    "user1": "mypassword",
}

class ChatRequest(BaseModel):
    message: str
    docs_dir: str | None = None

# --- Auth Helpers ---
def get_current_user(token: str = Depends(oauth2_scheme)):
    if token not in USERS_DB:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# --- Endpoints ---

@api.get("/")
async def serve_index():
    # Serves the HTML file directly
    return FileResponse('chatwindow.html')

@api.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_pw = USERS_DB.get(form_data.username)
    if not user_pw or form_data.password != user_pw:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    # Ensure user has a vault directory
    user_path = os.path.join(UPLOAD_ROOT, form_data.username)
    os.makedirs(user_path, exist_ok=True)
    
    return {"access_token": form_data.username, "token_type": "bearer"}

@api.get("/folders")
async def list_folders(user: str = Depends(get_current_user)):
    user_path = os.path.join(UPLOAD_ROOT, user)
    if not os.path.exists(user_path): 
        return []
    # Return list of directories
    return [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]

@api.post("/folders/{folder_name}")
async def create_folder(folder_name: str, user: str = Depends(get_current_user)):
    # Basic sanitization
    safe_name = "".join([c for c in folder_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_name:
        raise HTTPException(400, "Invalid folder name")
        
    folder_path = os.path.join(UPLOAD_ROOT, user, safe_name)
    os.makedirs(folder_path, exist_ok=True)
    return {"status": "created", "path": folder_path}

@api.post("/upload/{folder_name}")
async def upload_files_endpoint(folder_name: str, files: List[UploadFile] = File(...), user: str = Depends(get_current_user)):
    user_path = os.path.join(UPLOAD_ROOT, user, folder_name)
    if not os.path.exists(user_path):
        raise HTTPException(404, "Folder not found")

    saved_files = []
    for file in files:
        file_location = os.path.join(user_path, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        saved_files.append(file.filename)
    
    # Force re-ingest so the new file is immediately available in chat
    ingest_docs_impl(user_path, force=True)
        
    return {"status": "success", "files": saved_files}

@api.get("/files/{folder_name}")
async def list_files(folder_name: str, user: str = Depends(get_current_user)):
    user_path = os.path.join(UPLOAD_ROOT, user, folder_name)
    if not os.path.exists(user_path):
        return []
    return [f for f in os.listdir(user_path) if os.path.isfile(os.path.join(user_path, f))]

@api.delete("/files/{folder_name}/{file_name}")
async def delete_file(folder_name: str, file_name: str, user: str = Depends(get_current_user)):
    file_path = os.path.join(UPLOAD_ROOT, user, folder_name, file_name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        user_folder_path = os.path.join(UPLOAD_ROOT, user, folder_name)
        ingest_docs_impl(user_folder_path, force=True)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/chat")
def chat(req: ChatRequest, user: str = Depends(get_current_user)):
    # 1. Determine Target Directory
    # If the frontend sends a specific folder, use it.
    # If not, use the user's root vault directory.
    if req.docs_dir:
        target_dir = req.docs_dir
    else:
        target_dir = os.path.join(UPLOAD_ROOT, user)

    # 2. Ingest
    # This will now combine [target_dir] + [DEFAULT_DOCS_DIR] automatically
    result = ingest_docs_impl(target_dir, force=False)
    
    if not result.get("ok", False):
        return {"answer": f"Error ingesting docs: {result.get('error')}"}

    # 3. Use Router
    answer = main_agent_router(req.message)
    return {"answer": answer}

# Mount MCP (Optional)
api.mount("/mcp", mcp_app)