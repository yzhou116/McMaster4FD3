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
MIN_SIMILARITY = 0.02  # tune 0.01–0.05

# OCR
ENABLE_OCR = True
OCR_MAX_PDF_PAGES = 20
OCR_DPI = 600  # 400–600 recommended for small text/tables

# Debug toggles (keep false for MCP)
DEBUG_TOP_CHUNKS = False
DEBUG_OCR_TO_STDOUT = False

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


@dataclass
class AppState:
    docs_dir: str = DEFAULT_DOCS_DIR
    files: Dict[str, str] = field(default_factory=dict)
    index: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, str] = field(default_factory=dict)
    is_ready: bool = False


STATE = AppState()

mcp = FastMCP("LocalDocs-OCR-RAG")


def assert_tesseract_ready() -> None:
    """
    Ensure Tesseract is usable. If not, fail fast with a clear error.
    """
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "") or ""
    if cmd and os.path.exists(cmd):
        return

    if shutil.which("tesseract"):
        return

    raise RuntimeError(
        "Tesseract not found by Python.\n"
        "Fix by either:\n"
        "1) Install Tesseract and add it to PATH, OR\n"
        r"2) Set TESSERACT_CMD in .env to: C:\Program Files\Tesseract-OCR\tesseract.exe\n"
        f"Current pytesseract cmd: {cmd!r}"
    )


# -----------------------------
# Native extraction
# -----------------------------

def read_txt_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_docx_text(path: str) -> str:
    doc = Document(path)
    parts: List[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        t = (page.extract_text() or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


# -----------------------------
# OCR helpers
# -----------------------------

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    OCR-friendly preprocessing WITHOUT harsh thresholding.
    Thresholding can destroy colored text (emails in blue/red).
    """
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
    """
    More reliable for extracting emails than free-form OCR output.
    Uses Output.DICT (no pandas dependency).
    """
    img = preprocess_for_ocr(img)
    try:
        data = pytesseract.image_to_data(
            img,
            config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
            output_type=pytesseract.Output.DICT,
        )
        hits: List[str] = []
        for t in data.get("text", []) or []:
            if not t:
                continue
            hits.extend(EMAIL_RE.findall(str(t)))
        return sorted(set(hits))
    except Exception:
        return []


def render_pdf_page(page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def band_crops(img: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    Scan multiple horizontal bands to catch tables wherever they sit.
    """
    w, h = img.size
    bands = [
        ("band_10_40", (0, int(h * 0.10), w, int(h * 0.40))),
        ("band_20_55", (0, int(h * 0.20), w, int(h * 0.55))),
        ("band_25_65", (0, int(h * 0.25), w, int(h * 0.65))),
        ("band_35_80", (0, int(h * 0.35), w, int(h * 0.80))),
    ]
    return [(name, img.crop(box)) for name, box in bands]


def ocr_pdf_pages(path: str) -> str:
    """
    Robust PDF OCR:
    - Render each page at OCR_DPI
    - OCR full page + band crops
    - Include extracted email hits as separate blocks
    """
    if not ENABLE_OCR:
        return ""

    out_lines: List[str] = []
    try:
        doc = fitz.open(path)
        page_count = min(len(doc), OCR_MAX_PDF_PAGES)

        for pi in range(page_count):
            page = doc.load_page(pi)
            full = render_pdf_page(page, OCR_DPI)

            # Full-page OCR
            t_full = ocr_text(full, psm=6)
            emails_full = ocr_emails_via_data(full)

            if t_full:
                out_lines.append(f"[PDF_PAGE_FULL_OCR page={pi+1}]\n{t_full}")
            if emails_full:
                out_lines.append(f"[PDF_PAGE_FULL_EMAILS page={pi+1}]\n" + "\n".join(emails_full))

            # Band OCR (helps tables)
            for bname, crop in band_crops(full):
                t6 = ocr_text(crop, psm=6)
                t11 = ocr_text(crop, psm=11)
                merged = "\n".join([x for x in [t6, t11] if x]).strip()

                emails_band = ocr_emails_via_data(crop)
                if emails_band:
                    out_lines.append(f"[PDF_{bname}_EMAILS page={pi+1}]\n" + "\n".join(emails_band))

                # Only keep meaningful band text (avoid garbage spam)
                if len(merged) >= 60:
                    out_lines.append(f"[PDF_{bname}_OCR page={pi+1}]\n{merged}")

        doc.close()
    except Exception as e:
        out_lines.append(f"[PDF_OCR_ERROR] {e}")

    return "\n\n".join(out_lines).strip()


def ocr_docx_images(path: str) -> str:
    """
    Extract embedded images from DOCX and OCR them (best-effort).
    """
    if not ENABLE_OCR:
        return ""

    out: List[str] = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            media_files = [n for n in z.namelist() if n.startswith("word/media/")]
            for idx, name in enumerate(media_files):
                data = z.read(name)
                try:
                    img = Image.open(io.BytesIO(data))
                    t = ocr_text(img, psm=6)
                    if t:
                        out.append(f"[DOCX_OCR_IMAGE {idx+1} - {os.path.basename(name)}]\n{t}")
                except Exception:
                    continue
    except Exception:
        return ""

    return "\n\n".join(out).strip()


def extract_text(path: str) -> str:
    """
    Return combined text:
    - native extracted text
    - plus OCR text for images inside docx/pdf when enabled
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".md"):
        return read_txt_md(path)

    if ext == ".docx":
        native = read_docx_text(path)
        ocr = ocr_docx_images(path)
        return "\n\n".join([t for t in [native, ocr] if t]).strip()

    if ext == ".pdf":
        native = read_pdf_text(path)
        ocr = ocr_pdf_pages(path)
        combined = "\n\n".join([t for t in [native, ocr] if t]).strip()

        if DEBUG_OCR_TO_STDOUT and combined:
            print("\n[DEBUG] OCR for PDF:", os.path.basename(path))
            print(combined[:2000], "\n---(truncated)---\n")

        return combined

    return ""


# -----------------------------
# Indexing / Retrieval
# -----------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if size <= 0:
        return [text]
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap

    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i: i + size])
        i += step
    return chunks


def search_local_files(docs_dir: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    patterns = [
        os.path.join(docs_dir, "*.txt"),
        os.path.join(docs_dir, "*.md"),
        os.path.join(docs_dir, "*.docx"),
        os.path.join(docs_dir, "*.pdf"),
    ]

    for pat in patterns:
        for path in glob.glob(pat):
            name = os.path.basename(path)
            if name.startswith("~$"):
                continue
            try:
                text = extract_text(path)
                if text:
                    files[name] = text
            except Exception as e:
                print(f"Error reading {path}: {e}")

    return files


def build_index(files: Dict[str, str]) -> Dict[str, Any]:
    chunk_texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for filename, text in files.items():
        for idx, ch in enumerate(chunk_text(text)):
            if ch.strip():
                chunk_texts.append(ch)
                metas.append({"filename": filename, "chunk": idx, "text": ch})

    vec_word = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        analyzer="word",
    )
    X_word = vec_word.fit_transform(chunk_texts)

    vec_char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
    )
    X_char = vec_char.fit_transform(chunk_texts)

    return {
        "metas": metas,
        "vec_word": vec_word,
        "X_word": X_word,
        "vec_char": vec_char,
        "X_char": X_char,
        "files": files,
    }


def retrieve(index: Dict[str, Any], query: str, top_k: int = TOP_K_CHUNKS) -> List[Tuple[float, Dict[str, Any]]]:
    q_word = index["vec_word"].transform([query])
    q_char = index["vec_char"].transform([query])

    sim_word = cosine_similarity(q_word, index["X_word"]).flatten()
    sim_char = cosine_similarity(q_char, index["X_char"]).flatten()

    sim = 0.6 * sim_word + 0.4 * sim_char
    top_idx = sim.argsort()[::-1][:top_k]
    return [(float(sim[i]), index["metas"][i]) for i in top_idx]


# -----------------------------
# Conversation memory + cheap local shortcuts
# -----------------------------

def make_query_with_memory(user_q: str, memory: Dict[str, str]) -> str:
    q = user_q.strip()
    q_low = q.lower()

    has_email = "@" in q
    has_course = bool(re.search(r"\b4fd3\b", q_low))
    has_names = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", q))

    if has_email or has_course or has_names:
        return q

    extras: List[str] = []
    if memory.get("last_course"):
        extras.append(memory["last_course"])
    if memory.get("last_entities"):
        extras.append(memory["last_entities"])

    if extras:
        return q + " (" + ", ".join(extras) + ")"
    return q


def update_memory_from_answer(question: str, answer: str, memory: Dict[str, str]) -> None:
    q_low = question.lower()
    a = (answer or "").strip()

    if "4fd3" in q_low or "4fd3" in a.lower():
        memory["last_course"] = "4FD3"

    names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", a)
    names = [n for n in names if n.lower() not in {"you", "rules", "context"}]
    if names:
        memory["last_entities"] = ", ".join(sorted(set(names))[:6])


def looks_like_email_question(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["email", "e-mail", "emails", "mail"])


def extract_emails_for_course(files: Dict[str, str], course_code: str) -> List[str]:
    hits: List[str] = []
    course_code = course_code.strip()

    for _, text in files.items():
        if course_code not in text:
            continue
        for m in re.finditer(re.escape(course_code), text):
            start = max(0, m.start() - 800)
            end = min(len(text), m.end() + 800)
            window = text[start:end]
            hits.extend(EMAIL_RE.findall(window))

    return sorted(set(hits))


def local_short_circuit(question: str, index: Dict[str, Any], memory: Dict[str, str]) -> Optional[str]:
    if looks_like_email_question(question):
        q_low = question.lower()
        course = "4FD3" if "4fd3" in q_low else memory.get("last_course")
        if course:
            emails = extract_emails_for_course(index["files"], course)
            if emails:
                return "\n".join(emails)
    return None


# -----------------------------
# Gemini answering
# -----------------------------

def answer_with_gemini(question: str, index: Dict[str, Any], memory: Dict[str, str]) -> str:
    if not index.get("metas"):
        return "can not find it"

    # Token-saver: try local shortcut first
    local = local_short_circuit(question, index, memory)
    if local is not None:
        update_memory_from_answer(question, local, memory)
        return local

    effective_q = make_query_with_memory(question, memory)
    retrieved = retrieve(index, effective_q, top_k=TOP_K_CHUNKS)

    best_sim = retrieved[0][0] if retrieved else 0.0
    if best_sim < MIN_SIMILARITY:
        return "can not find it"

    ctx_parts: List[str] = []
    for score, meta in retrieved:
        if score <= 0:
            continue
        snippet = (meta["text"] or "")[:1600]
        ctx_parts.append(
            f"FILE: {meta['filename']}\nCHUNK: {meta['chunk']}\nSCORE: {score:.3f}\n{snippet}"
        )

    context = "\n\n---\n\n".join(ctx_parts).strip()
    if not context:
        return "can not find it"

    prompt = f"""
You are a strict assistant.

Rules:
- Use ONLY the provided CONTEXT to answer.
- If the answer is not clearly present in the CONTEXT, reply with exactly: can not find it
- Do not add extra words when replying "can not find it".

CONTEXT:
{context}

QUESTION:
{question}
""".strip()

    try:
        client = genai.Client(api_key=API_KEY)
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        text = (resp.text or "").strip()
        if not text:
            return "can not find it"

        if text.strip().lower() in {"can not find it", "cannot find it", "can't find it"}:
            return "can not find it"

        update_memory_from_answer(question, text, memory)
        return text
    except Exception as e:
        return f"Error calling Gemini API: {e}"


# -----------------------------
# Core callable functions for HTTP backend (IMPORTANT)
# -----------------------------

def ingest_docs_impl(docs_dir: str = "my_docs") -> Dict[str, Any]:
    """
    Core (callable) implementation of ingest_docs.
    Do NOT decorate this with @mcp.tool().
    """
    abs_dir = docs_dir if os.path.isabs(docs_dir) else os.path.join(BASE_DIR, docs_dir)

    if not os.path.exists(abs_dir):
        return {"ok": False, "error": f"Folder not found: {abs_dir}"}

    if ENABLE_OCR:
        try:
            assert_tesseract_ready()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    files = search_local_files(abs_dir)
    index = build_index(files) if files else {"metas": []}

    STATE.docs_dir = abs_dir
    STATE.files = files
    STATE.index = index
    STATE.is_ready = True

    return {
        "ok": True,
        "docs_dir": abs_dir,
        "file_count": len(files),
        "files": sorted(list(files.keys())),
        "chunk_count": len(index.get("metas", [])),
    }


def ask_impl(question: str) -> str:
    """
    Core (callable) implementation of ask.
    Do NOT decorate this with @mcp.tool().
    """
    if not STATE.is_ready:
        result = ingest_docs_impl("my_docs")
        if not result.get("ok", False):
            return f"Error: {result.get('error', 'unknown ingest error')}"

    if not API_KEY:
        return "Error: GEMINI_API_KEY not found in .env"

    return answer_with_gemini(question, STATE.index, STATE.memory)


def list_docs_impl() -> List[str]:
    if not STATE.is_ready:
        ingest_docs_impl("my_docs")
    return sorted(list(STATE.files.keys()))


def get_doc_text_impl(filename: str, max_chars: int = 4000) -> str:
    if not STATE.is_ready:
        ingest_docs_impl("my_docs")
    text = STATE.files.get(filename, "")
    return (text or "")[:max_chars] if text else "can not find it"


# -----------------------------
# MCP tools (wrappers)
# -----------------------------

@mcp.tool()
def ingest_docs(docs_dir: str = "my_docs") -> Dict[str, Any]:
    """MCP tool wrapper for ingest_docs_impl."""
    return ingest_docs_impl(docs_dir)


@mcp.tool()
def list_docs() -> List[str]:
    """MCP tool wrapper for list_docs_impl."""
    return list_docs_impl()


@mcp.tool()
def ask(question: str) -> str:
    """MCP tool wrapper for ask_impl."""
    return ask_impl(question)


@mcp.tool()
def get_doc_text(filename: str, max_chars: int = 4000) -> str:
    """MCP tool wrapper for get_doc_text_impl."""
    return get_doc_text_impl(filename, max_chars=max_chars)


# -----------------------------
# Entrypoint
# -----------------------------

if __name__ == "__main__":
    # MCP stdio server
    mcp.run()
