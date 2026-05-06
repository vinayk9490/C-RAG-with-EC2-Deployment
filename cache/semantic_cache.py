import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEmbeddings


COLLECTION   = "semantic_cache"
VECTOR_SIZE  = 384   # all-MiniLM-L6-v2 output dimension


class SemanticCache:
    """
    Stores question→answer pairs as vectors in a dedicated Qdrant collection.
    On lookup, if a semantically similar question (cosine similarity >= threshold)
    was answered before, the cached answer is returned — skipping the full pipeline.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embeddings: HuggingFaceEmbeddings,
        threshold: float = 0.92,
    ):
        self.client     = qdrant_client
        self.embeddings = embeddings
        self.threshold  = threshold
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def get(self, question: str) -> str | None:
        """Return a cached answer if a similar question exists, else None."""
        vector   = self.embeddings.embed_query(question)
        response = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=1,
            score_threshold=self.threshold,
        )
        points = response.points
        if points:
            score  = points[0].score
            answer = points[0].payload["answer"]
            print(f"[SemanticCache] HIT  (similarity={score:.3f})")
            return answer
        print("[SemanticCache] MISS")
        return None

    def set(self, question: str, answer: str) -> None:
        """Store a new question→answer pair in the cache."""
        vector = self.embeddings.embed_query(question)
        self.client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"question": question, "answer": answer},
                )
            ],
        )
        print(f"[SemanticCache] Stored: '{question[:60]}...'")
