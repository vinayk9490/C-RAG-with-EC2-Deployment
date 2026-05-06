# C-RAG — Corrective Retrieval-Augmented Generation

A production-ready RAG pipeline built with **LangGraph**, **Groq (LLaMA 3.3-70B)**, **Qdrant**, and **PostgreSQL**. It goes beyond basic RAG by grading retrieved documents, falling back to web search when they are not relevant, and persisting full conversation history per user via a UUID-scoped Postgres checkpointer. Repeated or semantically similar questions are served from a vector cache — skipping the pipeline entirely.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    Semantic Cache                        │
│   Embed question → search Qdrant (cosine ≥ 0.92?)       │
│                                                          │
│   HIT ──────────────────────────────────► Return answer  │
│   MISS ─────────────────────────────────► RAG Pipeline   │
└─────────────────────────────────────────────────────────┘
      │ (cache miss)
      ▼
┌─────────────────────────────────────────────────────────┐
│                  LangGraph Pipeline                      │
│                                                          │
│  [retriever]                                             │
│   Qdrant (vector) + BM25 (keyword) → EnsembleRetriever  │
│        │                                                 │
│        ▼                                                 │
│  [valid_documents]                                       │
│   LLM grades each doc: relevant (yes/no)?               │
│        │                                                 │
│        ├── ≥ 30% relevant ──────► [generate]            │
│        │                              │                  │
│        └── < 30% relevant             │                  │
│               │                       │                  │
│               ▼                       │                  │
│        [rewrite_query]                │                  │
│         Rewrite for web search        │                  │
│               │                       │                  │
│               ▼                       │                  │
│        [web_search]                   │                  │
│         Tavily → append results       │                  │
│               │                       │                  │
│               └──────────────────────►│                  │
│                                       ▼                  │
│                              LLaMA 3.3-70B answer        │
└─────────────────────────────────────────────────────────┘
      │
      ▼
Store answer in Semantic Cache (Qdrant)
Store Q&A in Postgres under user's UUID
```

---

## Key Features

| Feature | Details |
|---|---|
| **Corrective RAG** | LLM grades every retrieved document; triggers web search if relevance < 30% |
| **Hybrid Retrieval** | BM25 (keyword) + Qdrant (vector) fused via EnsembleRetriever (50/50 weight) |
| **Semantic Caching** | Questions with cosine similarity ≥ 0.92 return a cached answer instantly |
| **Persistent Memory** | Full conversation history stored in Postgres, scoped to a UUID per user |
| **Web Search Fallback** | Tavily search triggered automatically when documents are insufficient |
| **Query Rewriting** | LLM rewrites the query before web search for better results |
| **Streamlit UI** | Chat interface with real-time pipeline status and session management |

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3-70B Versatile |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector DB | Qdrant (Docker) |
| Graph Orchestration | LangGraph (StateGraph) |
| Persistent Memory | PostgreSQL via `langgraph-checkpoint-postgres` |
| Semantic Cache | Qdrant (`semantic_cache` collection) |
| Web Search | Tavily |
| Document Loading | PyMuPDF |
| UI | Streamlit |

---

## Project Structure

```
C-RAG/
│
├── main.py                          # CLI entry point
├── streamlit_app.py                 # Streamlit web UI
├── nodejs.pdf                       # Source document
│
├── Ingestion_Retrieval_Pipeline/
│   └── ensemble.py                  # Qdrant ingestion + BM25 + EnsembleRetriever
│
├── tasks/
│   ├── direct_llm_call.py           # RAG answer generation (Groq)
│   ├── document_evaluator.py        # LLM document grader
│   ├── query_rewriter.py            # Query rewriter for web search
│   └── tavilysearch.py              # Tavily web search tool
│
├── memory/
│   └── postgres_memory.py           # PostgresSaver context manager
│
├── cache/
│   └── semantic_cache.py            # SemanticCache (Qdrant-backed)
│
├── .env                             # API keys and DB URI
└── requirements.txt
```

---

## Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- PostgreSQL (Postgres.app on Mac or any running instance)
- API keys: Groq, Tavily, LangChain (optional tracing)

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd C-RAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key          # optional
LANGCHAIN_API_KEY=your_langchain_api_key    # optional tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=C-RAG-Hybrid-Search
POSTGRES_URI=postgresql://your_user@localhost:5432/crag_memory
```

### 3. Start Qdrant via Docker

```bash
# First run — pulls the image and creates a container
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    --name qdrant \
    qdrant/qdrant

# Subsequent runs — just start the existing container
docker start qdrant
```

Verify Qdrant is running:
```bash
docker ps --filter "name=qdrant"
```

### 4. Create the Postgres database

```bash
createdb crag_memory
```

The three checkpoint tables (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`) and the `semantic_cache` Qdrant collection are created automatically on first run.

### 5. Ingest the PDF into Qdrant

```bash
python -m Ingestion_Retrieval_Pipeline.ensemble
```

This loads `nodejs.pdf`, splits it into 800-token chunks, embeds them with `all-MiniLM-L6-v2`, and upserts them into the `Node-JS` Qdrant collection.

---

## Running the Application

### Option A — Streamlit UI (recommended)

```bash
streamlit run streamlit_app.py --server.headless true
```

Open `http://localhost:8501` in your browser.

**How to use:**
1. Click **New** in the sidebar to create a session — a UUID is generated for you
2. Copy and save the UUID — paste it next time to resume your conversation
3. Type a question in the chat input
4. Watch the pipeline status panel tick through the nodes in real time
5. The answer shows its source: `Semantic Cache`, `Document RAG`, or `Web Search + RAG`

### Option B — CLI

```bash
python main.py
```

```
Enter your UUID (or press Enter to start a new session): 
New session created. Your UUID: f3a1c2d4-...

Enter your question: What is error handling in Node.js?
```

---

## How Each Component Works

### Hybrid Retrieval (EnsembleRetriever)

Two retrievers run in parallel and their results are fused:
- **BM25** — keyword-based, good at exact term matches
- **Qdrant vector search** — semantic, good at meaning-based matches

Both return top-10 results, weighted equally (0.5 / 0.5).

### Document Grader (valid_documents)

Each retrieved document is scored by LLaMA 3.3-70B with a structured output (`yes` / `no`). If fewer than 30% of documents are relevant, the query is handed off to the web search path.

### Query Rewriter

Before hitting Tavily, the original question is rewritten by the LLM to be more search-engine-friendly (concise, keyword-rich).

### Persistent Memory (PostgreSQL)

Every graph run is checkpointed to Postgres after each node. The `messages` field in `GraphState` uses `Annotated[List[dict], operator.add]` — meaning each run **appends** to the list rather than replacing it. Passing the same UUID in `config` on a future run restores the full history.

### Semantic Cache (Qdrant)

Before every pipeline run, the question is embedded and compared against previously answered questions stored in the `semantic_cache` Qdrant collection. If cosine similarity ≥ 0.92, the cached answer is returned immediately — no LLM call, no retrieval, no web search.

The cache is shared across all users and both the CLI and UI.

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for LLaMA inference |
| `TAVILY_API_KEY` | Yes | Tavily API key for web search |
| `POSTGRES_URI` | Yes | PostgreSQL connection string |
| `LANGCHAIN_API_KEY` | No | LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing (`true`/`false`) |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |

---

## Graph Diagram

The LangGraph state machine is exported as `c_rag_graph.png` in the project root on every run.

```
START → retriever → valid_documents → generate → END
                          │
                          └──(web search needed)──► rewrite_query → web_search → generate → END
```
