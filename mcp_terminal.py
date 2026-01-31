import os
import glob
import re
import io
import zipfile
import shutil
from typing import Dict, List, Tuple, Any, Optional

from dotenv import load_dotenv
from google import genai

# For .docx text
from docx import Document

# For .pdf text
from pypdf import PdfReader

# OCR + image handling
import pytesseract
from PIL import Image, ImageOps, ImageFilter

# PDF tools
import fitz  # pymupdf

# General retrieval: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Env / Config
# -----------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()

# Optional: set in .env as TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MY_DOCS = os.path.join(BASE_DIR, "my_docs")

# Chunking knobs
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Retrieval knobs
TOP_K_CHUNKS = 8
MIN_SIMILARITY = 0.02  # tune 0.01–0.05

# OCR knobs
ENABLE_OCR = True
OCR_MAX_PDF_PAGES = 20
OCR_DPI = 600  # 400–600 recommended for small table text

# Debug
DEBUG_TOP_CHUNKS = False
DEBUG_OCR_TO_STDOUT = False

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def assert_tesseract_ready() -> None:
    """
    Ensure Tesseract is usable.
    """
    # If user provided TESSERACT_CMD, use it.
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
# Extraction (native text)
# -----------------------------

def read_txt_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_docx_text(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
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
    """
    img = preprocess_for_ocr(img)
    try:
        df = pytesseract.image_to_data(
            img,
            config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
            output_type=pytesseract.Output.DATAFRAME,
        )
        df = df.dropna(subset=["text"])
        hits: List[str] = []
        for t in df["text"].astype(str).tolist():
            hits.extend(EMAIL_RE.findall(t))
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

    out = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            media_files = [n for n in z.namelist() if n.startswith("word/media/")]
            for idx, name in enumerate(media_files):
                data = z.read(name)
                try:
                    img = Image.open(io.BytesIO(data))
                    # two-pass OCR is overkill for docx images; keep it simple
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
    - native extracted text (txt/md/docx/pdf)
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


def search_local_files() -> Dict[str, str]:
    files: Dict[str, str] = {}

    patterns = [
        os.path.join(MY_DOCS, "*.txt"),
        os.path.join(MY_DOCS, "*.md"),
        os.path.join(MY_DOCS, "*.docx"),
        os.path.join(MY_DOCS, "*.pdf"),
    ]

    for pat in patterns:
        for path in glob.glob(pat):
            name = os.path.basename(path)
            if name.startswith("~$"):  # Word temp/lock file
                continue

            try:
                text = extract_text(path)
                if text:
                    files[name] = text
            except Exception as e:
                print(f"Error reading {path}: {e}")

    return files


# -----------------------------
# Indexing + Retrieval (General)
# -----------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if size <= 0:
        return [text]
    overlap = max(0, min(overlap, size - 1))
    step = size - overlap

    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i : i + size])
        i += step
    return chunks


def build_index(files: Dict[str, str]) -> Dict[str, Any]:
    chunk_texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for filename, text in files.items():
        for idx, ch in enumerate(chunk_text(text)):
            if ch.strip():
                chunk_texts.append(ch)
                metas.append({"filename": filename, "chunk": idx, "text": ch})

    # Word ngrams
    vec_word = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        analyzer="word",
    )
    X_word = vec_word.fit_transform(chunk_texts)

    # Char ngrams (robust to OCR noise)
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
        "files": files,  # keep full texts for cheap regex lookups
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
# Conversational carryover
# -----------------------------

def make_query_with_memory(user_q: str, memory: Dict[str, str]) -> str:
    q = user_q.strip()
    q_low = q.lower()

    has_email = "@" in q
    has_course = bool(re.search(r"\b4fd3\b", q_low))
    has_names = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", q))  # "First Last"

    if has_email or has_course or has_names:
        return q

    extras = []
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


# -----------------------------
# Cheap local extractors (reduce Gemini token usage)
# -----------------------------

def looks_like_email_question(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["email", "e-mail", "mail", "emails"])


def extract_emails_for_course(files: Dict[str, str], course_code: str) -> List[str]:
    """
    Very cheap: find emails near the course code in any document.
    """
    hits: List[str] = []
    course_code = course_code.strip()

    for _, text in files.items():
        if course_code not in text:
            continue

        # window around first occurrence(s)
        for m in re.finditer(re.escape(course_code), text):
            start = max(0, m.start() - 800)
            end = min(len(text), m.end() + 800)
            window = text[start:end]
            hits.extend(EMAIL_RE.findall(window))

    return sorted(set(hits))


def local_short_circuit(question: str, index: Dict[str, Any], memory: Dict[str, str]) -> Optional[str]:
    """
    Answer some queries without Gemini to save tokens (but still general overall).
    """
    if looks_like_email_question(question):
        course = None
        q_low = question.lower()
        if "4fd3" in q_low:
            course = "4FD3"
        elif memory.get("last_course"):
            course = memory["last_course"]

        if course:
            emails = extract_emails_for_course(index["files"], course)
            if emails:
                return "\n".join(emails)

    return None


# -----------------------------
# Chat (Gemini)
# -----------------------------

def chat_with_gemini(question: str, index: Dict[str, Any], memory: Dict[str, str]) -> str:
    if not index["metas"]:
        return "can not find it"

    # Try cheap local answer first (saves tokens)
    local = local_short_circuit(question, index, memory)
    if local is not None:
        update_memory_from_answer(question, local, memory)
        return local

    effective_q = make_query_with_memory(question, memory)
    retrieved = retrieve(index, effective_q, top_k=TOP_K_CHUNKS)

    best_sim = retrieved[0][0] if retrieved else 0.0
    if best_sim < MIN_SIMILARITY:
        return "can not find it"

    ctx_parts = []
    for score, meta in retrieved:
        if score <= 0:
            continue
        ctx_parts.append(
            f"FILE: {meta['filename']}\nCHUNK: {meta['chunk']}\nSCORE: {score:.3f}\n{meta['text']}"
        )

    context = "\n\n---\n\n".join(ctx_parts).strip()
    if not context:
        return "can not find it"

    if DEBUG_TOP_CHUNKS:
        print("\n[DEBUG] Effective query:", effective_q)
        print("[DEBUG] Top retrieved:")
        for score, meta in retrieved[:4]:
            print(f"  - {meta['filename']} chunk {meta['chunk']} score {score:.3f}")

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


def main():
    print("--- Local File Chatbot (Gemini + TXT/MD/DOCX/PDF + OCR + TFIDF) ---")

    if not API_KEY:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return

    if not os.path.exists(MY_DOCS):
        print(f"Folder not found at: {MY_DOCS}")
        return

    # Ensure OCR is actually usable (fail fast with a clear message)
    if ENABLE_OCR:
        assert_tesseract_ready()

    files = search_local_files()
    print(f"Found {len(files)} readable files: {list(files.keys())}")

    index = build_index(files)
    print(f"Indexed {len(index['metas'])} chunks.")

    memory: Dict[str, str] = {}

    while True:
        try:
            q = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        answer = chat_with_gemini(q, index, memory)
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
