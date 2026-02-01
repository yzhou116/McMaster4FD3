import os
import shutil
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List
from server import mcp, STATE, ingest_docs_impl, ask_impl

mcp_app = mcp.http_app(path="/")
api = FastAPI(lifespan=mcp_app.lifespan)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_ROOT = os.path.join(os.getcwd(), "user_vaults")
os.makedirs(UPLOAD_ROOT, exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
USERS_DB = {
    "admin": "password123",
    "user1": "mypassword",
}

def get_current_user(token: str = Depends(oauth2_scheme)):
    if token not in USERS_DB:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

@api.get("/")
async def serve_index():
    return FileResponse('chatwindow.html')

@api.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_pw = USERS_DB.get(form_data.username)
    if not user_pw or form_data.password != user_pw:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": form_data.username, "token_type": "bearer"}

@api.get("/folders")
async def list_folders(user: str = Depends(get_current_user)):
    user_path = os.path.join(UPLOAD_ROOT, user)
    if not os.path.exists(user_path): return []