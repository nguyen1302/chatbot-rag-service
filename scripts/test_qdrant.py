from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=7366)

def list_by_doc_id(doc_id: str):
    result = client.scroll(
        collection_name="docs",
        scroll_filter=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
        ),
        limit=10
    )
    return [pt.payload["text"][:100] for pt in result[0]]  # Trả về 100 ký tự đầu mỗi chunk

if __name__ == "__main__":
    import sys
    doc_id = sys.argv[1] if len(sys.argv) > 1 else "product"
    results = list_by_doc_id(doc_id)
    for i, chunk in enumerate(results):
        print(f"[{i}] {chunk}\n")
