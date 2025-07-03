from app.services.intent_classifier import classify, check_question_followup,embed_question,retrieve_top_chunks
from typing import Tuple, List
from app.models.rag import RAGResponse

def build_prompt_with_context(question: str) -> RAGResponse:
    """
    Tạo prompt có context nếu intent là 'internal' hoặc 'hybrid'.
    Trả về đối tượng RAGResponse gồm:
        - prompt: prompt cuối cùng đưa vào LLM
        - context_chunks: danh sách đoạn nội dung liên quan
    """
    intent = classify(question)

    if intent in ["internal", "hybrid"]:
        # Kiểm tra xem đây có phải là câu hỏi follow-up không (log để theo dõi)
        if check_question_followup(question):
            print("Câu hỏi follow-up")

        # Embed câu hỏi và truy xuất các đoạn văn bản gần nhất
        vector = embed_question(question)
        top_chunks = retrieve_top_chunks(vector, top_k=3)

        # Lấy nội dung text từ các chunk (dạng dict hoặc string)
        context_chunks = [chunk['content'] if isinstance(chunk, dict) else chunk for chunk in top_chunks]

        # Ghép các đoạn context lại
        context_text = "\n".join(context_chunks)

        # Tạo prompt chuẩn
        prompt = f"""[Context liên quan]:
        {context_text}

        [Câu hỏi người dùng]:
        {question}"""

        # Trả về dạng RAGResponse
        return RAGResponse(prompt=prompt, context_chunks=context_chunks)

    # Nếu không cần context thì trả prompt gốc và context rỗng
    return RAGResponse(prompt=question, context_chunks=[])