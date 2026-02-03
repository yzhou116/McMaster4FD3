import os
import glob
import re
import io
import zipfile
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

from dotenv import load_dotenv
from google import genai

from docx import Document
from pypdf import PdfReader

import pytesseract
from PIL import Image, ImageOps, ImageFilter
import fitz  # pymupdf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastmcp import FastMCP
import pandas as pd

# -----------------------------
# Config / State
# -----------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOCS_DIR = os.path.join(BASE_DIR, "my_docs")

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Retrieval
TOP_K_CHUNKS = 8
MIN_SIMILARITY = 0.02

# OCR
ENABLE_OCR = True 
OCR_MAX_PDF_PAGES = 20
OCR_DPI = 600

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

@dataclass
class AppState:
    # We now track a LIST of directories
    scanned_dirs: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)
    index: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, str] = field(default_factory=dict)
    is_ready: bool = False

STATE = AppState()

mcp = FastMCP("LocalDocs-OCR-RAG")

# -----------------------------
# Helpers: Tesseract / OCR / Extraction
# -----------------------------

def assert_tesseract_ready() -> None:
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "") or ""
    if cmd and os.path.exists(cmd):
        return
    if shutil.which("tesseract"):
        return
    raise RuntimeError("Tesseract not found. Check TESSERACT_CMD in .env")

def read_txt_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_docx_text(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()]).strip()

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([(page.extract_text() or "").strip() for page in reader.pages]).strip()

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img

def ocr_text(img: Image.Image, psm: int) -> str:
    img = preprocess_for_ocr(img)
    cfg = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    return (pytesseract.image_to_string(img, config=cfg) or "").strip()

def ocr_emails_via_data(img: Image.Image) -> List[str]:
    img = preprocess_for_ocr(img)
    try:
        data = pytesseract.image_to_data(
            img,
            config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
            output_type=pytesseract.Output.DICT,
        )
        hits = []
        for t in data.get("text", []) or []:
            if t: hits.extend(EMAIL_RE.findall(str(t)))
        return sorted(set(hits))
    except Exception:
        return []

def render_pdf_page(page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))

def ocr_pdf_pages(path: str) -> str:
    if not ENABLE_OCR: return ""
    out_lines = []
    try:
        doc = fitz.open(path)
        page_count = min(len(doc), OCR_MAX_PDF_PAGES)
        for pi in range(page_count):
            page = doc.load_page(pi)
            full = render_pdf_page(page, OCR_DPI)
            t_full = ocr_text(full, psm=6)
            emails = ocr_emails_via_data(full)
            if t_full: out_lines.append(f"[PAGE {pi+1} OCR]\n{t_full}")
            if emails: out_lines.append(f"[PAGE {pi+1} EMAILS]\n" + "\n".join(emails))
        doc.close()
    except Exception as e:
        out_lines.append(f"[OCR ERROR] {e}")
    return "\n\n".join(out_lines).strip()

def ocr_docx_images(path: str) -> str:
    if not ENABLE_OCR: return ""
    out = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if name.startswith("word/media/"):
                    try:
                        img = Image.open(io.BytesIO(z.read(name)))
                        t = ocr_text(img, psm=6)
                        if t: out.append(f"[IMG {name}]\n{t}")
                    except: continue
    except: return ""
    return "\n\n".join(out).strip()

def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"): return read_txt_md(path)
    if ext == ".docx": return read_docx_text(path) + "\n" + ocr_docx_images(path)
    if ext == ".pdf": return read_pdf_text(path) + "\n" + ocr_pdf_pages(path)
    return ""

# -----------------------------
# Indexing & RAG
# -----------------------------

def chunk_text(text: str) -> List[str]:
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunks.append(text[i: i + CHUNK_SIZE])
    return chunks

def search_local_files(directories: List[str]) -> Dict[str, str]:
    """
    Scans multiple directories and aggregates the files.
    """
    files = {}
    for folder in directories:
        if not os.path.exists(folder):
            continue
            
        for ext in ["*.txt", "*.md", "*.docx", "*.pdf"]:
            # glob.glob does not support multiple dirs directly, so we loop
            for path in glob.glob(os.path.join(folder, ext)):
                if os.path.basename(path).startswith("~$"): continue
                try:
                    t = extract_text(path)
                    if t: 
                        # Use filename as key. Warning: Duplicate filenames in different folders will overwrite!
                        files[os.path.basename(path)] = t
                except Exception as e:
                    print(f"Err {path}: {e}")
    return files

def build_index(files: Dict[str, str]) -> Dict[str, Any]:
    chunk_texts = []
    metas = []
    for fn, txt in files.items():
        for idx, ch in enumerate(chunk_text(txt)):
            if ch.strip():
                chunk_texts.append(ch)
                metas.append({"filename": fn, "chunk": idx, "text": ch})
    
    if not chunk_texts: return {"metas": []}

    vec_word = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    X_word = vec_word.fit_transform(chunk_texts)
    
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))
    X_char = vec_char.fit_transform(chunk_texts)
    
    return {"metas": metas, "vec_word": vec_word, "X_word": X_word, 
            "vec_char": vec_char, "X_char": X_char, "files": files}

def retrieve(index: Dict[str, Any], query: str, top_k=TOP_K_CHUNKS):
    if not index.get("metas"): return []
    q_word = index["vec_word"].transform([query])
    q_char = index["vec_char"].transform([query])
    sim = (0.6 * cosine_similarity(q_word, index["X_word"]).flatten() + 
           0.4 * cosine_similarity(q_char, index["X_char"]).flatten())
    top_idx = sim.argsort()[::-1][:top_k]
    return [(float(sim[i]), index["metas"][i]) for i in top_idx]

# -----------------------------
# Smart Router & Agents
# -----------------------------

def call_gemini_simple(prompt: str) -> str:
    try:
        client = genai.Client(api_key=API_KEY)
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def query_excel_impl(query: str) -> str:
    """
    Scans all directories in STATE.scanned_dirs for Excel files.
    """
    if not STATE.scanned_dirs:
        return "No directories scanned yet."

    xlsx_files = []
    for folder in STATE.scanned_dirs:
        if os.path.exists(folder):
            found = glob.glob(os.path.join(folder, "*.xlsx"))
            xlsx_files.extend(found)
    
    # FILTER: Remove temporary open files (starts with ~$)
    xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith("~$")]

    if not xlsx_files:
        return "No Excel files found in any active folders."

    results = []
    
    # 2. Read headers/rows
    for f in xlsx_files:
        try:
            df = pd.read_excel(f)
            df_preview = df.head(10).to_markdown(index=False)
            results.append(f"FILE: {os.path.basename(f)}\n{df_preview}")
        except Exception as e:
            print(f"Error reading excel {f}: {e}")
            results.append(f"Error reading {os.path.basename(f)}")

    if not results:
         return "Could not read any Excel files."

    context = "\n\n".join(results)

    # 3. Ask Gemini
    prompt = f"""
    You are a data analyst. 

    Rules:
    1. You MUST first think step-by-step inside <thinking> tags. 
    2. If the answer is not in the Excel data, simply say "No information found in Excel."

    DATA:
    {context}
    
    QUESTION:
    {query}
    """
    return call_gemini_simple(prompt)

def clean_ocr_emails(email_list: List[str]) -> List[str]:
    valid = []
    for email in email_list:
        if "@" not in email or len(email) < 5: continue
        # Fix OCR typos
        email = email.replace("ssrinvw", "ssriniv")
        valid.append(email)
    return sorted(list(set(valid)))

def answer_with_gemini(question: str, index: Dict[str, Any], memory: Dict[str, str]) -> str:
    # 1. Check local shortcuts (emails)
    if "email" in question.lower():
        course = "4FD3" # simplifying for demo
        emails = []
        for fn, txt in index["files"].items():
            if course in txt:
                emails.extend(EMAIL_RE.findall(txt))
        if emails:
            # clean = clean_ocr_emails(emails)
            return f"<thinking>Found emails via regex scan.</thinking>\nHere are the emails:\n" + "\n".join(emails)

    # 2. RAG Retrieval
    retrieved = retrieve(index, question)
    if not retrieved or retrieved[0][0] < MIN_SIMILARITY:
        return "can not find it"

    context_str = "\n\n".join([f"source: {m['filename']}\n{m['text']}" for s, m in retrieved if s > 0])
    
    prompt = f"""
    You are a strict assistant.
    
    Rules:
    1. You MUST first think step-by-step inside <thinking> tags. Analyze the context.
    2. Then answer based ONLY on the context.
    3. If the context doesn't have the answer, say "can not find it".
    
    CONTEXT:
    {context_str}
    
    QUESTION: {question}
    """
    return call_gemini_simple(prompt)

def main_agent_router(user_query: str) -> str:
    """
    Routes to Excel or Docs, with Fallback.
    """
    router_prompt = f"""
    Classify this query into one tool:
    1. EXCEL_TOOL (tables, numbers, lists, financial)
    2. DOCS_TOOL (text, theories, policies, names)
    
    Query: "{user_query}"
    Reply only with EXCEL_TOOL or DOCS_TOOL.
    """
    decision = call_gemini_simple(router_prompt).upper()
    print(f"[ROUTER] Decision: {decision}")

    if "EXCEL" in decision:
        ans = query_excel_impl(user_query)
        # Fallback check
        if any(x in ans.lower() for x in ["no information", "cannot find", "not present"]):
            print("[ROUTER] Excel failed, switching to Docs...")
            return answer_with_gemini(user_query, STATE.index, STATE.memory)
        return ans
    
    return answer_with_gemini(user_query, STATE.index, STATE.memory)

# -----------------------------
# Core Ingest Logic
# -----------------------------

def ingest_docs_impl(user_target_dir: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    Ingests files from BOTH the global 'my_docs' AND the specific 'user_target_dir'.
    """
    # 1. Always include the Global Defaults
    dirs_to_scan = [DEFAULT_DOCS_DIR]
    
    # 2. If a user folder is provided, add it to the scan list
    if user_target_dir:
        abs_user_dir = user_target_dir if os.path.isabs(user_target_dir) else os.path.join(BASE_DIR, user_target_dir)
        if os.path.exists(abs_user_dir) and abs_user_dir != DEFAULT_DOCS_DIR:
            dirs_to_scan.append(abs_user_dir)

    # --- CACHE CHECK ---
    # We check if the set of directories we are about to scan is exactly what we have in memory
    # We sort them to ensure order doesn't matter for the check
    current_set = sorted(STATE.scanned_dirs)
    target_set = sorted(dirs_to_scan)
    
    if not force and STATE.is_ready and current_set == target_set:
        return {"ok": True, "cached": True, "count": len(STATE.files)}
    # -------------------

    if ENABLE_OCR:
        try:
            assert_tesseract_ready()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # Scan ALL directories
    files = search_local_files(dirs_to_scan)
    index = build_index(files) if files else {"metas": []}

    STATE.scanned_dirs = dirs_to_scan
    STATE.files = files
    STATE.index = index
    STATE.is_ready = True

    return {
        "ok": True,
        "scanned_dirs": dirs_to_scan,
        "file_count": len(files),
        "files": sorted(list(files.keys())),
        "chunk_count": len(index.get("metas", [])),
    }

def ask_impl(q: str):
    if not STATE.is_ready: ingest_docs_impl() # Default ingest if nothing ready
    return answer_with_gemini(q, STATE.index, STATE.memory)

# -----------------------------
# MCP Wrappers
# -----------------------------
@mcp.tool()
def ingest(path: str): return ingest_docs_impl(path)

@mcp.tool()
def ask(q: str): return ask_impl(q)

if __name__ == "__main__":
    mcp.run()