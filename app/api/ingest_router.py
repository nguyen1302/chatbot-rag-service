# app/api/ingest_router.py
from fastapi import APIRouter
from app.services.ingest_service import ingest_folder
from app.services.ingest_service import list_document_ids
from app.services.ingest_service import get_chunks_by_doc_id
from app.services.ingest_service import count_chunks_by_doc_id
from app.services.ingest_service import get_full_chunks_by_doc_id

router = APIRouter()

@router.post("/ingest")
async def ingest_all():
    ingest_folder("raw-data", force=True)
    return {"status": "ok", "message": "All PDFs or txt ingested"}

@router.get("/ingest/docs")
async def list_docs():
    return {"documents": list_document_ids()}

@router.get("/ingest/docs/{doc_id}")
async def get_doc_chunks(doc_id: str):
    chunks = get_chunks_by_doc_id(doc_id)
    return {
        "document_id": doc_id,
        "total_chunks": len(chunks),
        "chunks": chunks[:5]  # chỉ trả 5 đoạn đầu để tránh quá tải
    }
    
@router.get("/ingest/count/{doc_id}")
async def count_vectors(doc_id: str):
    total = count_chunks_by_doc_id(doc_id)
    return {"document_id": doc_id, "total_chunks": total}

@router.get("/ingest/full-chunks/{doc_id}")
async def get_full_chunks(doc_id: str, limit: int = 10, offset: int = 0):
    data = get_full_chunks_by_doc_id(doc_id, limit=limit, offset=offset)
    return {"document_id": doc_id, "chunks": data}
