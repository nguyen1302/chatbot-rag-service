from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import os

client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)

COLLECTION_NAME = "docs"

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

print(f"Collection `{COLLECTION_NAME}` created.")
