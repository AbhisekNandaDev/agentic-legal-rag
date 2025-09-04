# AgenticRAG Legal Assistant

A console-based, India-focused **agentic legal assistant** that combines **Retrieval-Augmented Generation (RAG)** over local PDFs with **web research** and a lightweight **agent graph** to produce grounded, cited answers. It supports hybrid retrieval (FAISS + BM25), optional OCR for scanned PDFs during indexing, and a REPL-style chat interface.

> Example banner you’ll see on run:
>
> ```text
> 👩‍⚖️  Lawyer chatbot ready. Ask about your rights. Type 'exit' to quit.
> ```

---

## ✨ Features

- **Agentic flow with LangGraph**: multi-node graph (intake → retrieval → web research → analysis → citation → final).
- **Hybrid retrieval**: semantic search via **FAISS** + lexical BM25 via **langchain_community.BM25Retriever**.
- **Local corpus indexing**: `embedd.py` ingests PDFs (with optional OCR) and stores vector index.
- **Web research**: Tavily-powered tool to complement local knowledge with fresh, India-focused results.
- **Citations & traceability**: responses include references to local docs and/or web sources.
- **Safety**: simple pre-analysis guardrails and loop guard to prevent runaway reasoning.
- **Console REPL**: run locally without any UI dependencies.

---

## 🗂️ Project Structure

```
AgenticRag_Legal_Assistant/
├─ data/
│  ├─ Constitution_of_India.pdf
│  └─ penal_code.pdf
├─ embedd.py          # PDF → chunks → FAISS index (with optional OCR)
├─ graph_app.py       # Agent graph definition + REPL entrypoint
├─ tools.py           # Tooling: web research, logging helpers, utilities
├─ requirements.txt
└─ .env               # API keys (OPENAI_API_KEY, TAVILY_API_KEY)
```

> **Note:** The indexing CLI and the chat app are in separate files (`embedd.py` and `graph_app.py`).

---

## 🔧 Prerequisites

- **Python 3.10+**
- **Tesseract OCR** (only if you want OCR on scanned PDFs during embedding)
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: Install from the official binaries and ensure `tesseract` is on PATH.
- **Poppler** (for PDF to image conversion when OCR is enabled)
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
  - Windows: Install Poppler and add `pdftoppm`/`pdftocairo` to PATH.

---

## 🚀 Quickstart

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r AgenticRag_Legal_Assistant/requirements.txt
```

### 3) Configure environment

Create/update `AgenticRag_Legal_Assistant/.env`:

```env
OPENAI_API_KEY="YOUR_OPENAI_KEY"
TAVILY_API_KEY="YOUR_TAVILY_KEY"  # optional but recommended for web research
```

> The code uses `langchain_openai.ChatOpenAI` for LLM calls. You can also set `OPENAI_API_KEY` in your shell environment instead of `.env`.

### 4) Prepare your knowledge base (index the PDFs)

Place your PDFs into `AgenticRag_Legal_Assistant/data/` (or pass explicit paths), then run:

```bash
python AgenticRag_Legal_Assistant/embedd.py   --pdf AgenticRag_Legal_Assistant/data/Constitution_of_India.pdf   --pdf AgenticRag_Legal_Assistant/data/penal_code.pdf   --out AgenticRag_Legal_Assistant/index.faiss   --ocr   --semantic-threshold 0.35   --maxmin-size 600,1200
```

**Flags (from `embedd.py`):**
- `--pdf`: one or more PDF paths to ingest
- `--out`: path to FAISS index file to create/overwrite
- `--ocr`: use OCR for image‑based/scanned PDFs
- `--semantic-threshold`: score cutoff to drop low‑quality chunks
- `--maxmin-size`: chunk size bounds as `min,max` (e.g., `600,1200`)

What it does:
1. Loads each PDF (via `pdfplumber`; if `--ocr`, via `pdf2image` + `pytesseract`)  
2. Cleans & chunks text to reasonable windows  
3. Embeds with **sentence-transformers** and builds a **FAISS** index  
4. Saves the index to `--out`

> You can re-run indexing anytime as you add/modify PDFs.

### 5) Run the agentic chatbot

```bash
python AgenticRag_Legal_Assistant/graph_app.py   --index AgenticRag_Legal_Assistant/index.faiss   --topk 6   --bm25-k 8   --no-web  # omit this flag if you want web research enabled
```

You’ll enter an interactive prompt. Type your legal question; type `exit` to quit.

Common flags (from `graph_app.py`):
- `--index`: path to FAISS index (from step 4)
- `--topk`: how many semantic neighbors to retrieve from FAISS
- `--bm25-k`: how many candidates to fetch via BM25
- `--web/--no-web`: enable/disable web research step
- `--temperature`: LLM creativity (defaults conservative)
- `--max-loops`: safety loop guard for the analysis node

---

## 🤖 Agent Graph (High-Level)

 ┌─────────┐
 │ Intake  │
 └────┬────┘
      │
      ▼
 ┌───────────────┐
 │ Retrieval     │
 │ (FAISS+BM25)  │
 └────┬────┬─────┘
      │    │
      │    └─ insufficient local grounding
      │
      ▼
 ┌───────────────┐
 │ Web Research  │
 │   (Tavily)    │
 └────┬──────────┘
      │
      ▼
 ┌─────────┐
 │ Analysis│
 └────┬────┘
      │
      ▼
 ┌───────────────┐
 │ Citation      │
 └────┬──────────┘
      │
      ▼
 ┌───────────────┐
 │ Final Answer  │
 └───────────────┘



**Nodes overview (from `graph_app.py`):**

- **Intake**  
  Normalizes the user message, sets language/intent hints, applies simple safety pre-checks, and initializes state (including a `loop_count` with a hard stop via `--max-loops`).

- **Retrieval**  
  - **FAISS** semantic search over embeddings created by `embedd.py`.  
  - **BM25** lexical retriever via `langchain_community.retrievers.BM25Retriever`.  
  - Results are merged & de-duplicated; top‑K pushed to `state["faiss_docs"]`/`state["bm25_docs"]` and combined context.

- **Web Research (optional)**  
  - Uses **Tavily** to fetch fresh, India‑focused results when local hits are thin or the query is news‑like or statute‑version‑sensitive.  
  - Results are logged and made available for the analysis step.

- **Analysis**  
  - **ChatOpenAI** produces a grounded draft.  
  - Tries to prioritize local corpus; falls back to web sources selectively.  
  - A loop guard ensures the agent doesn’t over‑think indefinitely.

- **Citation**  
  - Formats references: local doc chunks (with page/section if available) and web URLs/titles.

- **Final**  
  - Presents the answer with bulletproofed disclaimers (not legal advice), and structured takeaways next steps.

---

## 🧰 Tools & Dependencies

**Core libraries:**
- `langgraph`, `langchain`, `langchain-community`, `langchain-openai`
- `faiss-cpu`, `rank-bm25`
- `tavily-python`, `langchain-tavily` (web research)
- `sentence-transformers`, `numpy`, `scikit-learn`
- `pdfplumber`, `pdf2image`, `pytesseract`, `pillow` (PDF/OCR)
- `nltk` (tokenization/stopwords)
- `rich` (console UX)

Install via the provided `requirements.txt`.

**Environment variables (`.env`):**
```env
OPENAI_API_KEY="..."
TAVILY_API_KEY="..."
```

---

## 🧪 Example Prompts

- *“What are my rights if I’m detained by the police under CrPC?”*  
- *“Summarize Article 21 of the Constitution of India with case citations.”*  
- *“Is Section 420 of IPC bailable? Provide relevant statutes and recent changes.”*

---

## 📐 Configuration Tips

- If your PDFs are **scanned images**, pass `--ocr` during indexing; verify Tesseract and Poppler are installed and on PATH.
- Tune retrieval combining via `--topk` (FAISS) and `--bm25-k` (BM25). For statute-heavy corpora, a slightly larger BM25 can help.
- If answers look too verbose/creative, lower `--temperature` in `graph_app.py` args.
- For strictly offline runs, add `--no-web`; for up-to-date law/news items, keep web research enabled and ensure `TAVILY_API_KEY` is set.

---

## 🛠️ Development

- Format/lint: use your preferred toolchain (e.g., `ruff`, `black`).
- Add new documents: drop PDFs into `data/` and re-run `embedd.py`.
- Add tools: extend `tools.py` with a new function and wire it into the graph in `graph_app.py` (mirroring how `web_research` is used).

---

## ⚠️ Disclaimer

This repository is for **research and educational purposes**. It does **not** provide legal advice. Always consult a qualified attorney for advice on your specific situation.

---

## 📥 Download

You can download this README directly:

- **[Download README.md](sandbox:/mnt/data/README.md)**

---

## 🏷️ Suggested Repository Name & Description

- **Repo name:** `agentic-legal-rag-india`
- **Description:** *Agentic legal assistant for India using hybrid RAG (FAISS + BM25), OCR-able PDF indexing, and optional web research via Tavily. Built with LangGraph + LangChain.*

