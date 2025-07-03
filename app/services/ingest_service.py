import os
import uuid
from typing import List
from tqdm import tqdm
import fitz  # PyMuPDF

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.embedder import embed_question

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def recreate_collection_if_needed():
    if not client.collection_exists(COLLECTION_NAME):
        print("Creating collection...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

def check_exists_by_doc_id(doc_id: str) -> bool:
    result, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
        limit=1
    )
    return len(result) > 0

def delete_existing_by_doc_id(doc_id: str):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
        )
    )

def read_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

def ingest_single_file(file_path: str, document_id: str, force: bool = True):
    if not force and check_exists_by_doc_id(document_id):
        print(f"✅ Skipping {document_id} (already ingested)")
        return

    print(f"🚀 Ingesting: {document_id}")
    delete_existing_by_doc_id(document_id)

    text = read_file(file_path)
    chunks = chunk_text(text)

    points = []
    for chunk in tqdm(chunks, desc=f"Embedding {document_id}"):
        vector = embed_question(chunk)
        if vector is None:
            print(f"[❌] Failed to embed chunk: {chunk[:50]}")
            continue
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, "document_id": document_id}
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Done: {document_id}")

def ingest_folder(folder_path: str, force: bool = True):
    recreate_collection_if_needed()
    files = [f for f in os.listdir(folder_path) if f.endswith((".pdf", ".txt"))]
    for f in sorted(files):
        file_path = os.path.join(folder_path, f)
        doc_id = os.path.splitext(f)[0]
        ingest_single_file(file_path, doc_id, force=force)

def list_document_ids() -> List[str]:
    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=None,
        with_payload=True,
        limit=10000,
    )

    doc_ids = set()
    for point in response:
        doc_id = point.payload.get("document_id")
        if doc_id:
            doc_ids.add(doc_id)

    return sorted(doc_ids)

def get_chunks_by_doc_id(doc_id: str) -> List[str]:
    response, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
        ),
        with_payload=True,
        limit=1000
    )
    return [point.payload.get("text") for point in response if "text" in point.payload]

def count_chunks_by_doc_id(doc_id: str) -> int:
    total = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
            ),
            limit=100,
            offset=offset
        )
        total += len(points)
        if offset is None:
            break

    return total

def get_full_chunks_by_doc_id(doc_id: str, limit: int = 10, offset: int = 0):
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
        ),
        with_payload=True,
        with_vectors=True,  # <<< THIS!
        limit=limit,
        offset=offset
    )
    return [
        {
            "id": point.id,
            "vector": point.vector,
            "payload": point.payload
        }
        for point in points
    ]

