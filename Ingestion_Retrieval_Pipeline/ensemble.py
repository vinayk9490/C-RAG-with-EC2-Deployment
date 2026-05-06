from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


COLLECTION_NAME = "Node-JS"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension

qdrant = QdrantClient(host="localhost", port=6333)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)


def ingest_documents():
    file_path = os.path.join(os.path.dirname(__file__), "..", "nodejs.pdf")
    loader = PyMuPDFLoader(file_path)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    texts = [t for t in text_splitter.split_documents(document) if t.page_content.strip()]

    if qdrant.collection_exists(COLLECTION_NAME):
        print("deleted qdrant collection")
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    points = []
    for i, doc in enumerate(texts):
        vector = embeddings.embed_query(doc.page_content)
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={"text": doc.page_content, "page": doc.metadata.get("page", 0)},
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Inserted {len(points)} chunks into Qdrant collection '{COLLECTION_NAME}'")


def _fetch_chunks_from_qdrant():
    total = qdrant.get_collection(COLLECTION_NAME).points_count
    results, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=total,
        with_payload=True,
        with_vectors=False,
    )
    return [Document(page_content=point.payload["text"]) for point in results]


def create_retrievers(chunks=None):
    vectorstore = QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="text",
    )

    if chunks is None:
        chunks = _fetch_chunks_from_qdrant()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore.as_retriever(search_kwargs={"k": 10})],
        weights=[0.5, 0.5],
    )
    return ensemble_retriever

if __name__ == "__main__":
    ingest_documents()