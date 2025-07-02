import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest, PointStruct, VectorParams, Distance

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "docs"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def retrieve_top_chunks(query_vector: List[float], top_k: int = 5) -> List[str]:
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
    return [hit.payload.get("text", "") for hit in search_result]

if __name__ == "__main__":
    from dotenv import load_dotenv
    from embedder import embed_question

    load_dotenv()

    vec = embed_question("LMS360 là gì?")
    chunks = retrieve_top_chunks(vec, top_k=3)
    print("Result:")
    for c in chunks:
        print("-", c)

