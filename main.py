from fastapi import FastAPI
from app.api import rag_router
from app.api import ingest_router

app = FastAPI()

app.include_router(rag_router.router, prefix="/chat")
app.include_router(ingest_router.router, prefix="/api")
