from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

res = client.count(collection_name="docs", exact=True)
print(f"✅ Total vectors in 'docs': {res.count}")
