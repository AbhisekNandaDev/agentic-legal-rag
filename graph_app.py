# file: graph_app.py
import os
from typing import TypedDict, List, Literal, Optional, Dict, Any
from dataclasses import dataclass

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

from tools import (
    load_faiss_retriever,
    web_search,
    extract_top_quotes,
    format_docs_for_context,
)

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()

# =========================
# State
# =========================
class ConversationState(TypedDict):
    messages: List[Dict[str, Any]]
    user_query: str
    rewritten_query: Optional[str]
    subqueries: List[str]
    targets: List[str]
    faiss_docs: List[Any]
    web_results: List[Dict[str, Any]]
    draft_answer: Optional[str]
    quotes: List[Dict[str, str]]
    next: Optional[Literal["intake","retrieval","web_research","analysis","citation","final"]]
    loop_count: int
    # human-in-the-loop
    awaiting_input: bool
    questions_for_user: Optional[str]
    skip_intake_once: bool   # don’t re-ask intake when resuming after clarification

@dataclass
class Settings:
    index_dir: str = "legal_db"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_k: int = 6
    quotes: int = 4
    max_loops: int = 4
    prefer_sources: tuple = ("penal_code","bns","crpc","evidence","bharatiya nyaya","bharatiya sakshya")
    pause_after_intake: bool = True  # <-- always pause for your answers after intake

SET = Settings()

lawyer_llm  = ChatOpenAI(model=SET.model, temperature=SET.temperature)
analyst_llm = ChatOpenAI(model=SET.model, temperature=0)
parser = StrOutputParser()

# =========================
# Prompts
# =========================
INTAKE_PROMPT = PromptTemplate.from_template(
    "You are a friendly Indian constitutional lawyer. "
    "Ask 2–4 concise clarification questions and then summarize the issue in one line. Be empathetic and clear.\n\n"
    "User said:\n{q}\n\nYour message:"
)

REWRITE_PROMPT = PromptTemplate.from_template(
    "Rewrite the user's issue into a crisp legal research query for Indian law. "
    "Include relevant Articles/Sections if obvious (e.g., 'IPC §375 Explanation 2', 'CrPC §41A').\n\n"
    "Issue:\n{q}\n\nLegal research query:"
)

PLANNER_PROMPT = PromptTemplate.from_template(
    "Given the legal research query, produce:\n"
    "- 1–3 sub-queries (bullets) to retrieve the most relevant constitutional provisions, penal/procedural sections, or cases.\n"
    "- One line starting with 'SECTIONS:' listing exact tokens to target (e.g., 375, 90, 41A, 438, 114A, 164A).\n\n"
    "Query:\n{q}\n\nOutput strictly in this format:\n"
    "- <subquery 1>\n- <subquery 2>\nSECTIONS: <comma-separated tokens>"
)

SYNTH_PROMPT = PromptTemplate.from_template(
    "You are a senior legal analyst. Use ONLY the provided context (statutes, procedure, and web snippets from Indian sources) to answer. "
    "If not clearly supported, say 'Insufficient support in current context.'\n\n"
    "Question:\n{question}\n\nContext:\n{context}\n\n"
    "Write a balanced legal analysis (4–7 lines) explicitly mentioning the section numbers you rely on "
    "(e.g., 'IPC §375 Explanation 2', 'CrPC §41A', 'Evidence Act §114A'). End with a one-line conclusion. "
    "Include this safety note at the end: 'This is not legal advice; consult a qualified lawyer.'"
)

SUFFICIENCY_PROMPT = PromptTemplate.from_template(
    "Return ONLY YES or NO: Is the context sufficient to answer the legal question with confidence?\n\n"
    "Question:\n{q}\n\nContext:\n{context}\n\nSufficient? (YES/NO):"
)

REFINER_PROMPT = PromptTemplate.from_template(
    "Improve this legal research query by adding any missing Indian section tokens or keywords likely to retrieve "
    "consent definitions and procedural safeguards (e.g., 375, 90, 41A, 438, 114A, 'consent', 'misconception').\n\n"
    "Original:\n{q}\n\nImproved:"
)

CLARIFY_PROMPT = PromptTemplate.from_template(
    "The context is insufficient. Ask 2–4 short, highly targeted follow-up questions to let you answer confidently. "
    "Use plain language. End with: 'Reply to these questions in one message; I will resume.'\n\nIssue:\n{q}\n\nYour questions:"
)

# =========================
# Pretty logging
# =========================
def log_node(name: str, content: str | None = None):
    console.print(Panel.fit(f"[bold cyan]Node:[/bold cyan] {name}", subtitle="agent flow", border_style="cyan"))
    if content:
        console.print(Markdown(content))
    console.print(Rule(style="grey50"))

def log_table(title: str, rows: List[List[str]]):
    table = Table(title=title, show_lines=True, header_style="bold magenta")
    if rows:
        for i in range(len(rows[0])): table.add_column(f"col_{i}")
        for r in rows: table.add_row(*[str(x) for x in r])
    else:
        table.add_column("info"); table.add_row("(empty)")
    console.print(table); console.print(Rule(style="grey50"))

# =========================
# Helpers
# =========================
def _prefer_sources_first(docs: List[Any], preferred_prefixes: tuple) -> List[Any]:
    pref, rest = [], []
    for d in docs:
        src = (d.metadata.get("source") or "").lower()
        if src.startswith(preferred_prefixes) or any(p in src for p in preferred_prefixes):
            pref.append(d)
        else:
            rest.append(d)
    return pref + rest

def _bm25_merge(gathered: List[Any], query: str) -> List[Any]:
    if not gathered:
        return []
    bm_docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in gathered]
    try:
        bm = BM25Retriever.from_documents(bm_docs)
        bm.k = min(10, len(bm_docs))
        hits = bm.get_relevant_documents(query)
        hit_texts = set(h.page_content for h in hits)
        intersect = [d for d in gathered if d.page_content in hit_texts]
        others = [d for d in gathered if d.page_content not in hit_texts]
        return intersect + others
    except Exception:
        return gathered

def _filter_docs_by_targets(docs: List[Any], targets: List[str]) -> List[Any]:
    if not targets or not docs:
        return docs
    tset = {t.lower() for t in targets}
    filt = []
    for d in docs:
        s = d.page_content.lower()
        if any(t in s for t in tset):
            filt.append(d)
    return filt or docs

def _has_target_hits(docs: List[Any], targets: List[str]) -> bool:
    if not targets or not docs:
        return False
    tset = {t.lower() for t in targets}
    for d in docs:
        s = d.page_content.lower()
        if any(t in s for t in tset):
            return True
    return False

# =========================
# Nodes
# =========================
def intake_node(state: ConversationState) -> ConversationState:
    # Skip intake if we're resuming from a clarification
    if state.get("skip_intake_once"):
        state["skip_intake_once"] = False
        log_node("intake (skipped)", "_Resuming without re-asking questions._")
        state["next"] = "retrieval"
        return state

    q = state["user_query"]
    msg = parser.invoke(lawyer_llm.invoke(INTAKE_PROMPT.format(q=q))).strip()

    state["messages"].append({"role": "ai", "content": msg})
    log_node("intake", msg)

    if SET.pause_after_intake:
        # Pause here to collect user's answers before retrieval
        state["questions_for_user"] = msg
        state["awaiting_input"] = True
        state["skip_intake_once"] = True   # don’t re-ask intake on resume
        state["next"] = "final"            # end this turn; CLI will prompt ↪
        return state

    state["next"] = "retrieval"
    return state

def retrieval_node(state: ConversationState) -> ConversationState:
    retriever = load_faiss_retriever(SET.index_dir, k=SET.top_k)

    rewritten = parser.invoke(analyst_llm.invoke(REWRITE_PROMPT.format(q=state["user_query"]))).strip()
    plan_raw = parser.invoke(analyst_llm.invoke(PLANNER_PROMPT.format(q=rewritten)))
    lines = [s.strip() for s in plan_raw.splitlines() if s.strip()]
    subqs = [s.lstrip("-• ").strip() for s in lines if not s.upper().startswith("SECTIONS:")]
    sec_line = next((s for s in lines if s.upper().startswith("SECTIONS:")), "SECTIONS:")
    targets = [t.strip() for t in sec_line.split(":", 1)[-1].split(",") if t.strip()]

    state["rewritten_query"] = rewritten
    state["subqueries"] = subqs
    state["targets"] = targets

    log_node(
        "retrieval",
        f"**Rewritten query:** {rewritten}\n\n**Sub-queries:**\n" +
        ("\n".join([f"- {s}" for s in subqs[:3]]) if subqs else "- (none)") +
        (f"\n\n**SECTIONS:** {', '.join(targets) if targets else '(none)'}")
    )

    # FAISS gather for main query + top subqueries
    gathered, seen = [], set()
    for rq in [rewritten] + subqs[:2]:
        docs = retriever.invoke(rq)
        for d in docs:
            key = (d.page_content[:160], d.metadata.get("source",""), d.metadata.get("type",""))
            if key not in seen:
                seen.add(key); gathered.append(d)

    # Prefer penal/procedure, then BM25 re-rank
    gathered = _prefer_sources_first(gathered, SET.prefer_sources)
    gathered = _bm25_merge(gathered, rewritten)

    state["faiss_docs"] = gathered
    state["next"] = "web_research"
    return state

def web_research_node(state: ConversationState) -> ConversationState:
    # India-only domain allowlist to reduce noise
    allow = [
        "indiankanoon.org","indiacode.nic.in","prsindia.org","legislative.gov.in",
        "barandbench.com","livelaw.in","scobserver.in","scconline.com","lawbeat.in"
    ]
    if not os.getenv("TAVILY_API_KEY"):
        state["web_results"] = []
        log_node("web_research", "_No TAVILY_API_KEY set — skipping live web search._")
        state["next"] = "analysis"
        return state

    q = state["rewritten_query"] or state["user_query"]
    seeds = [q] + state.get("subqueries", [])[:2]
    results = []
    for s in seeds:
        try:
            # Preferred (if your tools.web_search supports include_domains)
            results.extend(web_search(s, k=6, include_domains=allow))
        except TypeError:
            # Backward compatible: older web_search signature (no include_domains)
            results.extend(web_search(s, k=6))

    # Dedupe by URL or (title+snippet)
    seen, uniq = set(), []
    for r in results:
        key = r.get("url") or (r.get("title"), r.get("snippet"))
        if key in seen:
            continue
        seen.add(key); uniq.append(r)

    state["web_results"] = uniq[:8]
    rows = [[r.get("title","(no title)"), r.get("url",""), r.get("snippet","")[:80] + "..."] for r in state["web_results"]]
    log_table("web_research results (India-focused)", rows)
    state["next"] = "analysis"
    return state

def analysis_node(state: ConversationState) -> ConversationState:
    state["loop_count"] += 1
    if state["loop_count"] > SET.max_loops:
        log_node("analysis", f"_Loop guard reached (>{SET.max_loops}). Proceeding to citation/final._")
        state["next"] = "citation"; return state

    faiss_ctx = format_docs_for_context(state["faiss_docs"], max_chars=1800)
    web_ctx = "\n".join([f"[web] {x['title']} — {x['snippet']} ({x['url']})" for x in state["web_results"][:6]])
    context = (faiss_ctx + ("\n" + web_ctx if web_ctx else "")).strip()

    draft = parser.invoke(analyst_llm.invoke(SYNTH_PROMPT.format(question=state["user_query"], context=context))).strip()
    state["draft_answer"] = draft

    ok_llm = parser.invoke(analyst_llm.invoke(SUFFICIENCY_PROMPT.format(q=state["user_query"], context=context))).strip().upper()
    ok = "YES" if _has_target_hits(state["faiss_docs"], state.get("targets", [])) else ok_llm

    log_node("analysis", f"**Sufficiency (LLM/heuristic):** {ok_llm} / {ok}\n\n{draft}")

    if "YES" in ok:
        state["next"] = "citation"; return state

    # Pause & ask user focused questions
    questions = parser.invoke(analyst_llm.invoke(CLARIFY_PROMPT.format(q=state["user_query"]))).strip()
    state["questions_for_user"] = questions
    state["awaiting_input"] = True
    state["skip_intake_once"] = True   # don’t re-run intake after your reply

    msg = "### ❓ I need a bit more info to be confident\n" + questions
    state["messages"].append({"role": "ai", "content": msg})
    log_node("analysis -> pause", msg)

    state["next"] = "final"
    return state

def citation_node(state: ConversationState) -> ConversationState:
    biased_docs = _filter_docs_by_targets(state["faiss_docs"], state.get("targets", []))
    quotes = extract_top_quotes(state["user_query"], biased_docs, max_quotes=SET.quotes)
    state["quotes"] = quotes

    pretty = []
    pretty.append("### 🧠 Lawyer’s Analysis")
    pretty.append(state["draft_answer"] or "")
    pretty.append("\n### 🔍 Exact Lines from Your Knowledge Base")
    if not quotes:
        pretty.append("(no direct lines matched)")
    else:
        for i, q in enumerate(quotes, 1):
            pretty.append(f'{i}. "{q["text"]}" — {q["source"]} [{q["type"]}]')
    if state["web_results"]:
        pretty.append("\n### 🌐 Similar/Previous Case References (India)")
        for r in state["web_results"][:5]:
            pretty.append(f'- {r.get("title","(no title)")} — {r.get("url","")}')
    msg = "\n".join(pretty)

    state["messages"].append({"role": "ai", "content": msg})
    log_node("citation", msg)
    state["next"] = "final"
    return state

def final_node(state: ConversationState) -> ConversationState:
    log_node("final", "_End of this turn. Answer questions or ask another question. Type 'exit' to quit._")
    return state

# =========================
# Build graph
# =========================
def build_app():
    graph = StateGraph(ConversationState)
    graph.add_node("intake", intake_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("web_research", web_research_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("citation", citation_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("intake")
    graph.add_conditional_edges("intake",        lambda s: s["next"], {"retrieval": "retrieval", "final": "final"})
    graph.add_conditional_edges("retrieval",     lambda s: s["next"], {"web_research": "web_research"})
    graph.add_conditional_edges("web_research",  lambda s: s["next"], {"analysis": "analysis"})
    graph.add_conditional_edges("analysis",      lambda s: s["next"], {"retrieval": "retrieval", "citation": "citation", "final": "final"})
    graph.add_conditional_edges("citation",      lambda s: s["next"], {"final": "final"})

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# =========================
# CLI
# =========================
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment.")
    if not os.getenv("TAVILY_API_KEY"):
        console.print("[yellow]Note:[/yellow] No TAVILY_API_KEY set — India web search will be skipped.")

    app = build_app()
    state: ConversationState = {
        "messages": [],
        "user_query": "",
        "rewritten_query": None,
        "subqueries": [],
        "targets": [],
        "faiss_docs": [],
        "web_results": [],
        "draft_answer": None,
        "quotes": [],
        "next": "intake",
        "loop_count": 0,
        "awaiting_input": False,
        "questions_for_user": None,
        "skip_intake_once": False,
    }

    console.print(Panel.fit("👩‍⚖️  Lawyer chatbot ready. Ask about your rights. Type 'exit' to quit.", border_style="green"))

    while True:
        # If the agent paused to ask you questions, take your clarification now
        if state.get("awaiting_input"):
            user_reply = console.input("[bold cyan]↪ Your clarification: [/bold cyan]").strip()
            if not user_reply or user_reply.lower() in {"exit", "quit"}:
                break
            # Merge your clarification and resume directly at retrieval
            state["messages"].append({"role": "human", "content": user_reply})
            state["user_query"] = (state.get("user_query") or "") + f"\n\nClarification: {user_reply}"
            state["awaiting_input"] = False
            state["questions_for_user"] = None
            state["skip_intake_once"] = True
            state["next"] = "retrieval"
        else:
            user_q = console.input("[bold]> [/bold]").strip()
            if not user_q or user_q.lower() in {"exit", "quit"}:
                break
            state["user_query"] = user_q
            state["loop_count"] = 0
            state["next"] = "intake"

        # Stream FULL STATE each step
        for step in app.stream(
            state,
            config={"recursion_limit": 60, "configurable": {"thread_id": "cli"}},
            stream_mode="values",
        ):
            state = step

        # Print the last AI message
        msgs = state.get("messages", [])
        for m in reversed(msgs):
            if m.get("role") == "ai":
                console.print(Panel(Markdown(m["content"]), title="Response", border_style="purple"))
                console.print(Rule(style="grey50"))
                break

        if not state.get("awaiting_input"):
            state["next"] = "intake"

if __name__ == "__main__":
    main()
