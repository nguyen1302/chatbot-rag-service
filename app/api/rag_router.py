from fastapi import APIRouter
from app.models.rag import RAGRequest, RAGResponse
from app.services import intent_classifier
from app.services import prompt_builder
from app.services import question_follow
from app.services import openai_llm
from app.services import count_token
from app.services import embedder_qa


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

@router.post("/test/intent")
def test_final_intent(req: RAGRequest):
    question = get_last_user_question(req.messages)
    
    # Xây dựng prompt và context từ câu hỏi
    rag_response = prompt_builder.build_prompt_with_context(question,messages=req.messages)
    prompt = rag_response.prompt
    context_chunks = rag_response.context_chunks


    # Gọi OpenAI API
    # answer = openai_llm.call_openai_from_rag(req, rag_response)

    return {
        "question": question,
        "final_prompt": prompt,
        "context_chunks": context_chunks,
        # "answer": answer if isinstance(answer, str) else "".join(answer)  # Nếu stream thì nối lại
    }

@router.post("/test/multi-intent")
def test_all_user_questions(req: RAGRequest):
    responses = []

    for msg in req.messages:
        if msg.role == "user":
            rag_response = prompt_builder.build_prompt_with_context(msg.content, messages=req.messages)
            # answer = openai_llm.call_openai_from_rag(req, rag_response)

            responses.append({
                "question": msg.content,
                "final_prompt": rag_response.prompt,
                "context_chunks": rag_response.context_chunks,
                # "answer": answer if isinstance(answer, str) else "".join(answer)
            })

    return responses

@router.post("/token/input")
def calculate_prompt_tokens(req: RAGRequest):
    """
    API dùng để xây dựng prompt và tính số token đầu vào (prompt_tokens).
    Không gọi OpenAI, chỉ xây prompt và đếm.
    """
    # Lấy câu hỏi user cuối
    question = get_last_user_question(req.messages)

    # Build prompt và context (CHƯA call OpenAI)
    context = prompt_builder.build_prompt_with_context(question)

    # Tạo messages (giống như sẽ gửi vào OpenAI)
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý hỗ trợ giáo dục thông minh."},
        {"role": "user", "content": context.prompt}
    ]

    # Tính token ngay tại đây, không phụ thuộc RAGResponse
    prompt_tokens = count_token.count_prompt_tokens(messages, model=req.model)

    # Trả về chi tiết
    return {
        "question": question,
        "final_prompt": context.prompt,
        "context_chunks": context.context_chunks,
        "prompt_tokens": prompt_tokens
    }

@router.post("/rag-test")
def rag_test(request: RAGRequest):
    result = embedder_qa.process_rag_request(request)
    return result