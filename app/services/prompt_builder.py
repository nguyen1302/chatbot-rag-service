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

    followup_context = []
    is_followup = False
    context_chunks = []

    # ====== 1. Kiểm tra phản hồi kiểu "sai rồi", "chưa đúng"... ======
    is_keyword_followup_feedback, keyword_context_feedback = question_follow.is_followup_key_feedback(question, messages)

    if is_keyword_followup_feedback:
        # keyword_context_feedback là câu hỏi trước cần kiểm lại
        extracted_question = keyword_context_feedback
        intent = classify(extracted_question)

        if intent in ["internal", "hybrid"]:
            vector = embed_question(extracted_question)
            top_chunks = retrieve_top_chunks(vector, top_k=5)
            context_chunks = [chunk['content'] if isinstance(chunk, dict) else chunk for chunk in top_chunks]
        
        followup_context = [keyword_context_feedback]
        question = f"Thông tin trên chưa đúng, bạn hãy trả lời lại: {extracted_question}"
        is_followup = True

    else:
        # ====== 2. Không có feedback → xử lý bình thường ======
        intent = classify(question)

        is_keyword_followup, keyword_context = question_follow.is_followup_key(question, messages)
        embedding_context = []
        is_semantic_followup = False

        if messages:
            rag_result = embedder_qa.process_rag_request(
                RAGRequest(model="text-embedding-3-small", messages=messages)
            )
            is_semantic_followup = rag_result.get("is_followup", False)
            embedding_context = rag_result.get("followup_context", [])

        is_followup = is_keyword_followup or is_semantic_followup
        followup_context = keyword_context + embedding_context if is_followup else []

        if intent in ["internal", "hybrid"]:
            vector = embed_question(question)
            top_chunks = retrieve_top_chunks(vector, top_k=3)
            context_chunks = [chunk['content'] if isinstance(chunk, dict) else chunk for chunk in top_chunks]

    # ====== 3. Tạo final prompt ======
    prompt_parts = []
    if followup_context:
        prompt_parts.append("[Ngữ cảnh cuộc hội thoại trước đó]:\n" + "\n".join(followup_context))
    if context_chunks:
        prompt_parts.append("[Context liên quan từ hệ thống]:\n" + "\n".join(context_chunks))
    prompt_parts.append(f"[Câu hỏi người dùng]:\n{question}")
    
    final_prompt = "\n\n".join(prompt_parts)

    return RAGResponse(prompt=final_prompt, context_chunks=context_chunks), is_followup


