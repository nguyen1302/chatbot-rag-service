from app.services.intent_classifier import classify, check_question_followup, embed_question, retrieve_top_chunks
from typing import Tuple, List
from app.models.rag import RAGResponse, ChatMessage


def get_followup_context_from_messages(messages: List[ChatMessage], current_question: str) -> List[str]:
    """
    Duyệt ngược message để tìm các message trước đó hỗ trợ ngữ cảnh cho câu hỏi follow-up.
    Trả về list các message content dạng text (để ghép vào prompt).
    """
    context = []
    seen_current = False

    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip() == current_question.strip():
            seen_current = True
            continue

        if not seen_current:    
            continue

        # Lấy những message liên quan trước đó
        if msg.role in ["user", "assistant"]:
            context.append(f"[{msg.role.upper()}]: {msg.content}")

        # Giới hạn lấy tối đa 3 cặp hội thoại gần nhất
        if len(context) >= 6:  # 3 cặp user-assistant
            break

    return list(reversed(context))  # Đảo ngược lại đúng thứ tự thời gian
