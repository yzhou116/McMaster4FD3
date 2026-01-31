from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server import mcp, STATE, ingest_docs_impl, ask_impl
from server import list_docs_impl, get_doc_text_impl, ingest_docs_impl, STATE
# Optional MCP HTTP app (handy for debugging)
mcp_app = mcp.http_app(path="/")

api = FastAPI(lifespan=mcp_app.lifespan)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # dev only
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    docs_dir: str | None = None

@api.get("/health")
def health():
    return {"status": "ok"}

@api.get("/debug/ingest")
def debug_ingest():
    # Force ingest and return stats
    result = ingest_docs_impl("my_docs")
    return {
        "ingest_result": result,
        "is_ready": STATE.is_ready,
        "docs_dir": STATE.docs_dir,
        "file_count": len(STATE.files),
        "chunk_count": len(STATE.index.get("metas", [])) if STATE.index else None,
    }

@api.get("/debug/docs")
def debug_docs():
    return {
        "docs_dir": STATE.docs_dir,
        "docs": list_docs_impl(),
    }

@api.get("/debug/doc_text")
def debug_doc_text(filename: str, max_chars: int = 1200):
    return {
        "filename": filename,
        "preview": get_doc_text_impl(filename, max_chars=max_chars),
    }

@api.post("/chat")
def chat(req: ChatRequest):
    # Ingest docs if needed
    if req.docs_dir:
        result = ingest_docs_impl(req.docs_dir)
        if not result.get("ok", False):
            return {"answer": f"Error: {result.get('error', 'unknown ingest error')}"}
    elif not STATE.is_ready:
        result = ingest_docs_impl("my_docs")
        if not result.get("ok", False):
            return {"answer": f"Error: {result.get('error', 'unknown ingest error')}"}

    answer = ask_impl(req.message)
    return {"answer": answer}

# Optional: expose MCP endpoints at /mcp
api.mount("/mcp", mcp_app)
