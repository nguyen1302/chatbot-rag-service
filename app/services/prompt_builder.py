from app.services.intent_classifier import classify, check_question_followup, embed_question, retrieve_top_chunks
from typing import Tuple, List
from app.models.rag import RAGResponse, ChatMessage
from app.services.question_follow import get_followup_context_from_messages
from app.services import question_follow
from app.models.rag import RAGRequest
from app.services import embedder_qa  # Đảm bảo đã import đúng module process_rag_request

def build_prompt_with_context(question: str, messages: List[ChatMessage] = None) -> RAGResponse:
    """
    Tạo prompt có context dựa trên intent và follow-up context.
    """
    # Bước 1: Xác định intent
    intent = classify(question)

    # Bước 2: Kiểm tra follow-up theo từ khóa
    is_keyword_followup, keyword_context = question_follow.is_followup_key(question, messages)

    # Bước 3: Nếu có messages → xử lý follow-up context theo embedding
    embedding_context = []
    is_semantic_followup = False

    if messages:
        rag_result = embedder_qa.process_rag_request(
            RAGRequest(model="text-embedding-3-small", messages=messages)
        )
        is_semantic_followup = rag_result.get("is_followup", False)
        embedding_context = rag_result.get("followup_context", [])

    # Gộp context nếu có
    if is_semantic_followup or is_keyword_followup:
        is_followup= True
    else: is_followup=False
    followup_context = keyword_context + embedding_context if is_followup else []
    

    # Bước 4: Xử lý theo intent
    context_chunks = []
    if intent in ["internal", "hybrid"]:
        vector = embed_question(question)
        top_chunks = retrieve_top_chunks(vector, top_k=3)
        context_chunks = [chunk['content'] if isinstance(chunk, dict) else chunk for chunk in top_chunks]

    # Bước 5: Tạo prompt
    prompt_parts = []

    if followup_context:
        prompt_parts.append("[Ngữ cảnh cuộc hội thoại trước đó]:\n" + "\n".join(followup_context))

    if context_chunks:
        prompt_parts.append("[Context liên quan từ hệ thống]:\n" + "\n".join(context_chunks))

    prompt_parts.append(f"[Câu hỏi người dùng]:\n{question}")
    final_prompt = "\n\n".join(prompt_parts)

    return RAGResponse(prompt=final_prompt, context_chunks=context_chunks), is_followup

