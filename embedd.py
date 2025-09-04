# file: embedd.py
import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# --- PDF handling (text + OCR fallback) ---
def pdf_to_txt(pdf_path: str) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise SystemExit("Please install pdfplumber: pip install pdfplumber")

    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            texts.append(t)
    raw = "\n".join(texts)
    return raw


def ocr_pdf_to_txt(pdf_path: str, dpi: int = 300, lang: str = "eng") -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise SystemExit(
            "Install OCR deps: pip install pdf2image pytesseract pillow (plus system Tesseract & Poppler)."
        )
    pages = convert_from_path(pdf_path, dpi=dpi)
    out = []
    for img in pages:
        txt = pytesseract.image_to_string(img, lang=lang)
        out.append(txt)
    return "\n\n".join(out)


# --- Cleaning ---
def normalize_legal_text(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", s)  # de-hyphenate line breaks
    s = re.sub(r"\r", "", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


# --- Sentence tokenization ---
def tokenize_sentences(text: str) -> List[str]:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
    from nltk.tokenize import sent_tokenize
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    return sentences


# --- Semantic embeddings for chunking ---
def sentence_embeddings(sentences: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(sentences, convert_to_numpy=True)


# --- Semantic chunking (group consecutive similar sentences) ---
def semantic_chunking(
    sentences: List[str], embs: np.ndarray, threshold: float = 0.65, min_len: int = 2
) -> List[str]:
    if not sentences:
        return []
    chunks = []
    current_chunk = [sentences[0]]
    current_vec = embs[0]
    for i in range(1, len(sentences)):
        sim = 1 - cosine_distances([current_vec], [embs[i]])[0][0]
        # start a new chunk only if current one has reached min_len
        if sim < threshold and len(current_chunk) >= min_len:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_vec = embs[i]
        else:
            current_chunk.append(sentences[i])
            current_vec = (current_vec + embs[i]) / 2
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


# --- Max–Min chunking (simple coverage-oriented clustering of indices) ---
def max_min_chunking(
    sentences: List[str], embs: np.ndarray, chunk_size: int = 5
) -> List[str]:
    n = len(sentences)
    if n == 0:
        return []
    used = set()
    chunks = []
    while len(used) < n:
        # start from first unused index to keep order
        idx = next(i for i in range(n) if i not in used)
        cluster = [idx]
        used.add(idx)

        while len(cluster) < chunk_size and len(used) < n:
            dists = cosine_distances([embs[cluster[-1]]], embs)[0]
            candidates = [(i, d) for i, d in enumerate(dists) if i not in used]
            if not candidates:
                break
            next_idx = min(candidates, key=lambda x: x[1])[0]
            cluster.append(next_idx)
            used.add(next_idx)

        chunk_text = " ".join(sentences[i] for i in cluster)
        chunks.append(chunk_text)
    return chunks


# --- Build or UPDATE FAISS index (incremental) ---
def upsert_faiss(docs: List[Tuple[str, dict]], index_dir: str) -> None:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    os.makedirs(index_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings()  # requires OPENAI_API_KEY
    lc_docs = [Document(page_content=txt, metadata=meta) for txt, meta in docs]

    index_path = Path(index_dir) / "index.faiss"
    pkl_path = Path(index_dir) / "index.pkl"

    if index_path.exists() and pkl_path.exists():
        # Load existing index, add, and save back
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        vs.add_documents(lc_docs)
        vs.save_local(index_dir)
        print(f"🔁 Updated existing FAISS index at: {index_dir} (added {len(lc_docs)} docs)")
    else:
        # Create new index
        vs = FAISS.from_documents(lc_docs, embeddings)
        vs.save_local(index_dir)
        print(f"🆕 Created new FAISS index at: {index_dir} (stored {len(lc_docs)} docs)")


def process_pdf(pdf_path: str, use_ocr: bool, semantic_threshold: float, maxmin_size: int):
    print(f"→ Processing: {pdf_path}")
    # Extract text
    if use_ocr:
        print("   Using OCR to extract text...")
        raw = ocr_pdf_to_txt(pdf_path)
    else:
        print("   Extracting selectable text with pdfplumber...")
        raw = pdf_to_txt(pdf_path)

    text = normalize_legal_text(raw)
    sentences = tokenize_sentences(text)
    if not sentences:
        raise SystemExit(f"No sentences extracted from {pdf_path}. Try --ocr if scanned or check PDF quality.")

    print(f"   Extracted {len(sentences)} sentences")

    embs = sentence_embeddings(sentences)

    # Chunking (two strategies)
    semantic_chunks = semantic_chunking(sentences, embs, threshold=semantic_threshold, min_len=2)
    maxmin_chunks = max_min_chunking(sentences, embs, chunk_size=maxmin_size)

    print(f"   Semantic chunks: {len(semantic_chunks)}")
    print(f"   Max–Min chunks:  {len(maxmin_chunks)}")

    # Add source metadata for traceability
    src_name = Path(pdf_path).name
    docs = [(c, {"type": "semantic", "source": src_name}) for c in semantic_chunks] + \
           [(c, {"type": "maxmin",  "source": src_name}) for c in maxmin_chunks]

    return docs


def main():
    parser = argparse.ArgumentParser(description="PDF(s) -> Text -> Chunk -> Embed -> FAISS (incremental)")
    parser.add_argument(
        "--pdf",
        nargs="+",
        required=True,
        help="Path(s) to PDF(s). You can pass multiple PDFs: --pdf a.pdf b.pdf c.pdf",
    )
    parser.add_argument("--out", default="legal_db", help="FAISS index directory (will be created or updated)")
    parser.add_argument("--ocr", action="store_true", help="Use OCR (for scanned PDFs)")
    parser.add_argument("--semantic_threshold", type=float, default=0.65, help="Similarity threshold for semantic chunking")
    parser.add_argument("--maxmin_size", type=int, default=5, help="Sentences per Max–Min chunk")
    args = parser.parse_args()

    # Aggregate all docs from all PDFs, then upsert once (faster)
    all_docs: List[Tuple[str, dict]] = []
    for pdf in args.pdf:
        docs = process_pdf(pdf, use_ocr=args.ocr, semantic_threshold=args.semantic_threshold, maxmin_size=args.maxmin_size)
        all_docs.extend(docs)

    print(f"\n📦 Total new chunks to upsert: {len(all_docs)}")
    print(f"📁 Upserting into index: {args.out}")
    upsert_faiss(all_docs, args.out)
    print("✅ Done.")


if __name__ == "__main__":
    main()
