from fastapi import APIRouter
from app.models.rag import RAGRequest, RAGResponse
from app.services import intent_classifier

router = APIRouter()

# Helper để lấy câu hỏi cuối cùng từ user
def get_last_user_question(messages: list) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""

@router.post("/", response_model=RAGResponse)
def handle_rag(req: RAGRequest):
    question = get_last_user_question(req.messages)
    intent = intent_classifier.classify(question)

    if intent == "internal":
        # TODO: chạy embed + retrieve
        prompt = f"[INTERNAL MODE]\nCâu hỏi: {question}"
        return RAGResponse(prompt=prompt, context_chunks=["chunk1", "chunk2"])
    
    # external → gửi câu hỏi nguyên xi cho GPT xử lý
    return RAGResponse(prompt=question)

