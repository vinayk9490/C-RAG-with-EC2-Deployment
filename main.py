from dotenv import load_dotenv
load_dotenv()

import os
import operator
import uuid
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

from typing import Annotated, List, TypedDict
from langchain_core.documents import Document

from Ingestion_Retrieval_Pipeline.ensemble import create_retrievers
from tasks.document_evaluator import documents_evaluator
from tasks.direct_llm_call import llm_call, format_docs
from tasks.query_rewriter import rewrite_prompt
from tasks.tavilysearch import tavily_results

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = os.path.join(os.path.dirname(__file__), "nodejs.pdf")
loader    = PyMuPDFLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: str
    # each run appends {"role": ..., "content": ...} dicts — never overwritten
    messages: Annotated[List[dict], operator.add]


# retrieve the relevant documents for a user-query
def retriever(state: GraphState):
    user_query = state["question"]
    ensemble_retriever = create_retrievers(chunks)
    relevant_docs = ensemble_retriever.invoke(user_query)
    print(f"printing relevant documents from ensemble retriever")
    print(relevant_docs)
    return {"documents": relevant_docs}


# grade each retrieved document and keep only relevant ones
RELEVANCE_THRESHOLD = 0.30  # 95% of docs must be relevant to skip web search

def valid_documents(state: GraphState):
    docs = state["documents"]
    query = state["question"]

    filtered_docs = []
    for doc in docs:
        if not doc.page_content.strip():
            continue
        result = documents_evaluator(doc.page_content, query)
        if result == "yes": #indicates that document is relevant
            filtered_docs.append(doc)

    total = len([d for d in docs if d.page_content.strip()])
    relevant_pct = len(filtered_docs) / total if total > 0 else 0
    print(f"Relevant docs: {len(filtered_docs)}/{total} ({relevant_pct:.0%})")

    web_search_needed = "no" if relevant_pct >= RELEVANCE_THRESHOLD else "yes"
    return {"documents": filtered_docs, "web_search": web_search_needed}


# routing function: decides next node after valid_documents
def route_documents(state: GraphState):
    if state["web_search"] == "yes":
        return "rewrite_query"
    return "generate"


# rewrite the user query for a better web search
def rewrite_query(state: GraphState):
    question = state["question"]
    rewritten = rewrite_prompt(question)
    return {"question": rewritten}


# perform web search with the rewritten query
def web_search(state: GraphState):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    web_search_tool = tavily_results()
    raw = web_search_tool.invoke({"query": question[:400]})
    if "error" in raw:
        raise RuntimeError(f"Tavily search failed: {raw['error']}")
    docs = raw["results"]
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}


# generate final answer using the relevant docs
def generate(state: GraphState):
    docs = state["documents"]
    question = state["question"]
    formatted = format_docs(docs)
    generation = llm_call(formatted, question)
    new_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": generation},
    ]
    return {"generation": generation, "messages": new_messages}

from langgraph.graph import END, StateGraph, START
workflow = StateGraph(GraphState)

#creating the nodes & edges
workflow.add_node("retriever", retriever)           #document retrieval node
workflow.add_node("web_search", web_search)         #tavily search node
workflow.add_node("rewrite_query", rewrite_query)   #rewrite user-query node
workflow.add_node("generate", generate)             #direct rag output generator
workflow.add_node("valid_documents", valid_documents) #Grade documents node

workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "valid_documents")
workflow.add_conditional_edges(
    "valid_documents",
    route_documents,
    {
        "rewrite_query": "rewrite_query",
        "generate": "generate"
    }
)

workflow.add_edge("rewrite_query", "web_search")
workflow.add_edge("web_search","generate")
workflow.add_edge("generate", END)

from pprint import pprint
from memory import get_postgres_checkpointer

with get_postgres_checkpointer() as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)

    # --- user inputs ---
    thread_id = input("Enter your UUID (or press Enter to start a new session): ").strip()
    if not thread_id:
        thread_id = str(uuid.uuid4())
        print(f"New session created. Your UUID: {thread_id}")

    question = input("Enter your question: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    config = {"configurable": {"thread_id": thread_id}}

    # Load and display previous history for this UUID
    prior_state = app.get_state(config).values
    prior_messages = prior_state.get("messages", [])
    if prior_messages:
        print(f"\n--- Chat history for {thread_id} ---")
        for msg in prior_messages:
            label = "You" if msg["role"] == "user" else "Assistant"
            print(f"{label}: {msg['content']}")
        print("------------------------------------\n")

    inputs = {"question": question, "messages": []}
    for output in app.stream(inputs, config=config):
        for key, value in output.items():
            pprint(f"Node '{key}':")
        pprint("\n---\n")

    print("\nAnswer:", value["generation"])

    history = app.get_state(config).values.get("messages", [])
    print(f"\n[thread_id={thread_id}] Total messages stored in Postgres: {len(history)}")