import os
import uuid
from typing import List, Dict
from tqdm import tqdm
import fitz  # PyMuPDF
from pydantic import BaseModel
import re



from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.services.embedder import embed_question

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "docs"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 50

class MarkdownText(BaseModel):
    text: str
    chunk_size: int = 700
    chunk_overlap: int = 50

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


def chunk_text(text: str, section: str, subsection: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    return [f"{section} – {subsection} – ý {i+1}\n\n{chunk.strip()}" for i, chunk in enumerate(chunks)]

def ingest_single_file(file_path: str, document_id: str, force: bool = True):
    if not force and check_exists_by_doc_id(document_id):
        print(f"✅ Skipping {document_id} (already ingested)")
        return

    print(f"🚀 Ingesting: {document_id}")
    delete_existing_by_doc_id(document_id)

    raw_text = read_file(file_path)
    chunks = parse_markdown_and_chunk(raw_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    points = []
    for chunk in chunks:
        text = chunk["text"]
        chunk_id = chunk["chunk_id"]

        vector = embed_question(text)
        if vector is None:
            print(f"[❌] Failed to embed chunk: {text[:50]}")
            continue

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                "chunk_id": chunk_id,
                "document_id": document_id
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Done: {document_id}")


def ingest_folder(folder_path: str, force: bool = True):
    recreate_collection_if_needed()
    _ingest_recursive(folder_path, force, base_folder=folder_path)


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
        with_vectors=False,  # <<< THIS!
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

def _ingest_recursive(current_path: str, force: bool, base_folder: str):
    entries = os.listdir(current_path)
    files = [f for f in entries if os.path.isfile(os.path.join(current_path, f)) and f.endswith((".pdf", ".txt"))]
    folders = [f for f in entries if os.path.isdir(os.path.join(current_path, f))]

    if files and folders:
        raise ValueError(f"❌ Thư mục '{current_path}' chứa cả tệp và thư mục — không được phép!")

    if files:
        for f in sorted(files):
            file_path = os.path.join(current_path, f)
            # Tạo doc_id từ đường dẫn tương đối, thay dấu '/' bằng '_'
            relative_path = os.path.relpath(file_path, start=base_folder)
            doc_id = os.path.splitext(relative_path)[0].replace(os.sep, "_")
            ingest_single_file(file_path, doc_id, force=force)
        return

    for folder in sorted(folders):
        sub_path = os.path.join(current_path, folder)
        _ingest_recursive(sub_path, force, base_folder)


def parse_markdown_and_chunk(text: str,
                              chunk_size: int = 700,
                              chunk_overlap: int = 50) -> List[Dict[str, str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    def clean_content(raw: str) -> str:
        lines = raw.splitlines()
        filtered = [line for line in lines if line.strip() and not re.fullmatch(r'#*', line.strip())]
        return "\n".join(filtered)

    all_chunks = []

    # Tách theo các tiêu đề cấp # (main sections)
    top_sections = re.split(r'(?=^#\s+)', text.strip(), flags=re.MULTILINE)

    for top_section in top_sections:
        match_main_title = re.match(r'^#\s+(.+)', top_section.strip())
        if not match_main_title:
            continue

        main_title = match_main_title.group(1).strip() or "Untitled"
        body = top_section[len(match_main_title.group(0)):].strip()

        # Nếu không có ## nào → chunk toàn bộ body
        if not re.search(r'^##\s+', body, flags=re.MULTILINE):
            content = clean_content(body)
            chunks = splitter.split_text(content)
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": f"{main_title}\n{chunk.strip()}",
                    "chunk_id": f"{main_title} | chunk {idx}"
                })
            continue

        # Nếu có ## → chia tiếp theo section
        sections = re.split(r'(?=^##\s+)', body, flags=re.MULTILINE)

        # 👉 Chunk phần nằm giữa main_title và ## đầu tiên
        first_section_start = re.search(r'^##\s+', body, flags=re.MULTILINE)
        if first_section_start:
            intro_body = body[:first_section_start.start()].strip()
            intro_body = clean_content(intro_body)
            intro_chunks = splitter.split_text(intro_body)
            for idx, chunk in enumerate(intro_chunks):
                all_chunks.append({
                    "text": f"{main_title}\n{chunk.strip()}",
                    "chunk_id": f"{main_title} | intro | chunk {idx}"
                })

        for section in sections:
            match_section = re.match(r'^##\s+(.+)', section.strip())
            if not match_section:
                continue

            section_title = match_section.group(1).strip() or "Untitled"
            section_body = section[len(match_section.group(0)):].strip()
            section_body = clean_content(section_body)

            # Nếu không có ### → chunk toàn bộ section
            if not re.search(r'^###\s+', section_body, flags=re.MULTILINE):
                chunks = splitter.split_text(section_body)
                for idx, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": f"{main_title}\n{section_title}\n{chunk.strip()}",
                        "chunk_id": f"{main_title} | {section_title} | chunk {idx}"
                    })
                continue

            # Nếu có ### → chia theo subsection
            subsections = re.split(r'(?=^###\s+)', section_body, flags=re.MULTILINE)

            for subsec in subsections:
                match_sub = re.match(r'^###\s+(.+)', subsec.strip())
                if match_sub:
                    subsection_title = match_sub.group(1).strip() or "Untitled"
                    content = subsec[len(match_sub.group(0)):].strip()
                    content = clean_content(content)
                    chunks = splitter.split_text(content)
                    for idx, chunk in enumerate(chunks):
                        all_chunks.append({
                            "text": f"{main_title}\n{section_title}\n{subsection_title}\n{chunk.strip()}",
                            "chunk_id": f"{main_title} | {section_title} | {subsection_title} | chunk {idx}"
                        })
                else:
                    # orphan đoạn không có ### nhưng nằm sau section
                    content = clean_content(subsec)
                    chunks = splitter.split_text(content)
                    for idx, chunk in enumerate(chunks):
                        all_chunks.append({
                            "text": f"{main_title}\n{section_title}\n{chunk.strip()}",
                            "chunk_id": f"{main_title} | {section_title} | orphan-subsection | chunk {idx}"
                        })

    return all_chunks