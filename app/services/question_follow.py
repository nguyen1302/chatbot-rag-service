from app.services.intent_classifier import classify, check_question_followup, check_feedback_followup
from typing import Tuple, List
from app.models.rag import RAGResponse, ChatMessage, RAGRequest
from app.services import embedder_qa

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


def is_followup_key(
    question: str,
    messages: List[ChatMessage]
) -> Tuple[bool, List[str]]:
    """
    Chỉ kiểm tra follow-up theo từ khóa. Trả về (is_followup_by_keyword, followup_context)
    """
    is_keyword_followup = check_question_followup(question)

    if is_keyword_followup:
        for i in range(len(messages) - 1, 0, -1):
            if messages[i - 1].role == "user" and messages[i].role == "assistant":
                recent_qa = f"Q: {messages[i - 1].content} A: {messages[i].content}"
                return True, [recent_qa]
        return True, []  # fallback nếu không tìm được cặp Q-A

    return False, []

def is_followup_key_feedback(
    question: str,
    messages: List[ChatMessage]
) -> Tuple[bool, str]:
    """
    Chỉ kiểm tra follow-up theo từ khóa. Trả về (is_followup_by_keyword, followup_context_str)
    """
    is_keyword_followup = check_feedback_followup(question)

    if is_keyword_followup:
        for i in range(len(messages) - 1, 0, -1):
            if messages[i - 1].role == "user" and messages[i].role == "assistant":
                recent_qa = f"Q: {messages[i - 1].content} A: {messages[i].content}"
                return True, recent_qa
        return True, ""  # fallback nếu không tìm được cặp Q-A

    return False, ""
