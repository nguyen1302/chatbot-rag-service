from app.services.intent_classifier import classify, check_question_followup, embed_question, retrieve_top_chunks
from typing import Tuple, List
from app.models.rag import RAGResponse, ChatMessage
from app.services.question_follow import get_followup_context_from_messages

def build_prompt_with_context(question: str, messages: List[ChatMessage] = None) -> RAGResponse:
    """
    Tạo prompt có context dựa trên intent và follow-up context.
    
    Args:
        question: Câu hỏi hiện tại
        messages: Lịch sử chat để xác định context follow-up (optional)
    
    Returns:
        RAGResponse: Chứa prompt cuối cùng và context chunks
    """
    # Bước 1: Xác định intent
    intent = classify(question)
    
    # Bước 2: Kiểm tra có phải câu hỏi follow-up không
    is_followup = check_question_followup(question)
    
    # Bước 3: Lấy context từ lịch sử chat nếu là follow-up
    followup_context = []
    if is_followup and messages:
        followup_context = get_followup_context_from_messages(messages, question)
    
    # Bước 4: Xử lý theo intent
    if intent in ["internal", "hybrid"]:
        # Embed câu hỏi và truy xuất các đoạn văn bản gần nhất
        vector = embed_question(question)
        top_chunks = retrieve_top_chunks(vector, top_k=3)
        
        # Lấy nội dung text từ các chunk
        context_chunks = [chunk['content'] if isinstance(chunk, dict) else chunk for chunk in top_chunks]
        
        # Ghép các đoạn context từ database
        db_context = "\n".join(context_chunks)
        
        # Tạo prompt với đầy đủ context
        prompt_parts = []
        
        # Thêm ngữ cảnh follow-up nếu có
        if followup_context:
            followup_text = "\n".join(followup_context)
            prompt_parts.append(f"[Ngữ cảnh cuộc hội thoại trước đó]:\n{followup_text}")
        
        # Thêm context từ database
        if db_context:
            prompt_parts.append(f"[Context liên quan từ hệ thống]:\n{db_context}")
        
        # Thêm câu hỏi hiện tại
        prompt_parts.append(f"[Câu hỏi người dùng]:\n{question}")
        
        # Ghép tất cả lại
        final_prompt = "\n\n".join(prompt_parts)
        
        return RAGResponse(prompt=final_prompt, context_chunks=context_chunks)
    
    elif intent == "external":
        # Xử lý intent external: chỉ có ngữ cảnh follow-up + question
        prompt_parts = []
        
        # Thêm ngữ cảnh follow-up nếu có
        if followup_context:
            followup_text = "\n".join(followup_context)
            prompt_parts.append(f"[Ngữ cảnh cuộc hội thoại trước đó]:\n{followup_text}")
        
        # Thêm câu hỏi hiện tại
        prompt_parts.append(f"[Câu hỏi người dùng]:\n{question}")
        
        # Ghép tất cả lại
        final_prompt = "\n\n".join(prompt_parts)
        
        return RAGResponse(prompt=final_prompt, context_chunks=[])
    
    # Fallback case - trả về prompt gốc
    return RAGResponse(prompt=question, context_chunks=[])

