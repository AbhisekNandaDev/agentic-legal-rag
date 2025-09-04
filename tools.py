# file: tools.py
import os
import re
from typing import List, Dict, Any,Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch  # modern package

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

# =========================
# FAISS loader
# =========================
def load_faiss_retriever(index_dir: str, k: int = 6):
    embeddings = OpenAIEmbeddings()
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k, "fetch_k": max(32, k)})

# =========================
# Web search (Tavily, robust)
# =========================
# --- in tools.py, replace your web_search with this one ---

def web_search(query: str, k: int = 5, include_domains: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Tavily search with optional domain allowlist.
    Returns list[{title,url,snippet}], or [] if no key / error.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        tool = TavilySearch(max_results=k, include_answer=False, include_raw_content=False)
        payload = {"query": query}
        if include_domains:
            payload["include_domains"] = include_domains  # supported by Tavily
        resp = tool.invoke(payload) or {}
        results = resp.get("results", []) or []
    except Exception:
        return []
    out = []
    for r in results:
        if isinstance(r, dict):
            out.append({
                "title": r.get("title",""),
                "url": r.get("url",""),
                "snippet": (r.get("content","") or "")[:300],
            })
    return out


# =========================
# Quote extraction
# =========================
STOP = set("""
the is are am a an and or of to in on for by with as at that this those these be been was were it its from into than
then there their them such any all not no but if shall may can would should could under within without
""".split())

def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z\[\(0-9])", text)
    sents = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 600:
            sents.extend([x.strip() for x in p.split("; ") if x.strip()])
        else:
            sents.append(p)
    return sents

def _score(query: str, sent: str) -> float:
    qw = {w for w in re.findall(r"[A-Za-z]+", query.lower()) if w not in STOP}
    sw = {w for w in re.findall(r"[A-Za-z]+", sent.lower()) if w not in STOP}
    if not qw or not sw:
        return 0.0
    inter = len(qw & sw)
    score = inter / (len(qw | sw))
    if any(p in sent.lower() for p in [
        "article","section","clause","fundamental","rights","freedom","public order","reasonable restrictions",
        "consent","consensual","notice","summons","bail","41a","438","114a","375","90"
    ]):
        score += 0.05
    return score

def _boost_by_targets(sent: str, targets: List[str]) -> float:
    s = sent.lower()
    return sum(0.25 for t in targets if t and t.lower() in s)

def extract_top_quotes(
    query: str,
    docs: List[Any],
    max_quotes: int = 4,
    targets: List[str] | None = None
) -> List[Dict[str, str]]:
    targets = targets or []
    cands = []
    for d in docs:
        for s in _sentences(d.page_content):
            base = _score(query, s)
            boosted = base + _boost_by_targets(s, targets)
            if boosted > 0:
                cands.append((boosted, s.strip(), d.metadata.get("source","unknown"), d.metadata.get("type","unknown")))
    if not cands:
        for d in docs:
            sents = _sentences(d.page_content)
            if sents:
                cands.append((0.0001, sents[0].strip(), d.metadata.get("source","unknown"), d.metadata.get("type","unknown")))
    cands.sort(key=lambda x: x[0], reverse=True)
    out, seen = [], set()
    for _, text, src, typ in cands:
        key = (text, src, typ)
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "source": src, "type": typ})
        if len(out) >= max_quotes:
            break
    return out

# =========================
# Formatting
# =========================
def format_docs_for_context(docs: List[Any], max_chars: int = 2000) -> str:
    blobs, total = [], 0
    for i, d in enumerate(docs, 1):
        t = d.metadata.get("type", "unknown")
        src = d.metadata.get("source", "unknown")
        text = d.page_content.replace("\n", " ")
        snippet = text[: min(600, len(text))]
        line = f"[faiss:{i}|{t}|{src}] {snippet}..."
        total += len(line)
        if total > max_chars:
            break
        blobs.append(line)
    return "\n".join(blobs)
