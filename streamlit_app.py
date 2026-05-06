from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import operator
import streamlit as st
import psycopg
from psycopg.rows import dict_row
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.postgres import PostgresSaver

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

from Ingestion_Retrieval_Pipeline.ensemble import create_retrievers
from tasks.document_evaluator import documents_evaluator
from tasks.direct_llm_call import llm_call, format_docs
from tasks.query_rewriter import rewrite_prompt
from tasks.tavilysearch import tavily_results
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cache import SemanticCache

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="C-RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

# ── Graph state ───────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: str
    messages: Annotated[List[dict], operator.add]


RELEVANCE_THRESHOLD = 0.30

NODE_LABELS = {
    "retriever":       "Retrieving documents from Qdrant + BM25",
    "valid_documents": "Evaluating document relevance",
    "rewrite_query":   "Rewriting query for web search",
    "web_search":      "Searching the web via Tavily",
    "generate":        "Generating final answer",
}


# ── Semantic cache (shared across all sessions) ───────────────────────────────
@st.cache_resource(show_spinner="Initialising semantic cache...")
def build_semantic_cache() -> SemanticCache:
    qdrant     = QdrantClient(host="localhost", port=6333)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return SemanticCache(qdrant, embeddings, threshold=0.92)


# ── Build & cache the compiled graph once ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading PDF and building RAG pipeline...")
def build_app():
    # Load and chunk the PDF
    file_path = os.path.join(os.path.dirname(__file__), "nodejs.pdf")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200
    ).split_documents(PyMuPDFLoader(file_path).load())

    # ── Node definitions (chunks captured via closure) ────────────────────────
    def retriever(state: GraphState):
        relevant_docs = create_retrievers(chunks).invoke(state["question"])
        return {"documents": relevant_docs}

    def valid_documents(state: GraphState):
        filtered = []
        non_empty = [d for d in state["documents"] if d.page_content.strip()]
        for doc in non_empty:
            if documents_evaluator(doc.page_content, state["question"]) == "yes":
                filtered.append(doc)
        relevant_pct = len(filtered) / len(non_empty) if non_empty else 0
        return {
            "documents": filtered,
            "web_search": "no" if relevant_pct >= RELEVANCE_THRESHOLD else "yes",
        }

    def route_documents(state: GraphState):
        return "rewrite_query" if state["web_search"] == "yes" else "generate"

    def rewrite_query(state: GraphState):
        return {"question": rewrite_prompt(state["question"])}

    def web_search(state: GraphState):
        raw = tavily_results().invoke({"query": state["question"][:400]})
        if "error" in raw:
            raise RuntimeError(f"Tavily error: {raw['error']}")
        web_doc = Document(
            page_content="\n".join(d["content"] for d in raw["results"])
        )
        return {"documents": state["documents"] + [web_doc], "question": state["question"]}

    def generate(state: GraphState):
        generation = llm_call(format_docs(state["documents"]), state["question"])
        return {
            "generation": generation,
            "messages": [
                {"role": "user",      "content": state["question"]},
                {"role": "assistant", "content": generation},
            ],
        }

    # ── Build graph ───────────────────────────────────────────────────────────
    workflow = StateGraph(GraphState)
    workflow.add_node("retriever",       retriever)
    workflow.add_node("valid_documents", valid_documents)
    workflow.add_node("rewrite_query",   rewrite_query)
    workflow.add_node("web_search",      web_search)
    workflow.add_node("generate",        generate)

    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "valid_documents")
    workflow.add_conditional_edges(
        "valid_documents", route_documents,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search",    "generate")
    workflow.add_edge("generate",      END)

    # ── Postgres checkpointer (persistent connection for Streamlit lifetime) ──
    conn = psycopg.connect(
        os.environ["POSTGRES_URI"],
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()

    return workflow.compile(checkpointer=checkpointer)


# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("thread_id", ""),
    ("messages",  []),
    ("source",    ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

app            = build_app()
semantic_cache = build_semantic_cache()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("C-RAG Assistant")
    st.caption("Corrective RAG · LangGraph · Groq · Qdrant · Postgres")
    st.divider()

    st.subheader("Session")
    uuid_input = st.text_input(
        "Session UUID",
        value=st.session_state.thread_id,
        placeholder="Paste UUID to resume, or leave blank",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load", use_container_width=True, type="primary"):
            tid = uuid_input.strip()
            if not tid:
                st.error("Enter a UUID first.")
            else:
                config = {"configurable": {"thread_id": tid}}
                prior = app.get_state(config).values
                st.session_state.thread_id = tid
                st.session_state.messages  = prior.get("messages", [])
                st.session_state.source    = ""
                st.rerun()

    with col2:
        if st.button("New", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages  = []
            st.session_state.source    = ""
            st.rerun()

    if st.session_state.thread_id:
        st.divider()
        st.caption("Active UUID")
        st.code(st.session_state.thread_id, language=None)
        turns = len(st.session_state.messages) // 2
        st.caption(f"{turns} turn(s) stored in Postgres")

        if st.button("Clear history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("C-RAG Chat")

if not st.session_state.thread_id:
    st.info(
        "Use the sidebar to **Load** an existing session (paste a UUID) "
        "or click **New** to start a fresh one."
    )
    st.stop()

# Render stored messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask anything about Node.js…")

if question:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Show the user bubble immediately
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        # ── 1. Semantic cache check ───────────────────────────────────────────
        cached = semantic_cache.get(question)

        if cached:
            st.caption("Source: Semantic Cache (similar question answered before)")
            st.markdown(cached)
            st.session_state.messages.extend([
                {"role": "user",      "content": question},
                {"role": "assistant", "content": cached},
            ])

        else:
            # ── 2. Cache miss → run full RAG pipeline ─────────────────────────
            status_box = st.status("Running pipeline…", expanded=True)
            generation = None
            used_web   = False

            for output in app.stream(
                {"question": question, "messages": []},
                config=config,
            ):
                for node_name, node_value in output.items():
                    label = NODE_LABELS.get(node_name, node_name)
                    status_box.write(f"✅ {label}")

                    if node_name == "web_search":
                        used_web = True
                    if node_name == "generate" and "generation" in node_value:
                        generation = node_value["generation"]

            status_box.update(label="Pipeline complete", state="complete", expanded=False)

            if generation:
                source_tag = "Web Search + RAG" if used_web else "Document RAG"
                st.caption(f"Source: {source_tag}")
                st.markdown(generation)

                # Store in semantic cache for future similar questions
                semantic_cache.set(question, generation)

                st.session_state.messages.extend([
                    {"role": "user",      "content": question},
                    {"role": "assistant", "content": generation},
                ])
