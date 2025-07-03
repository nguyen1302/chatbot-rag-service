import os
from openai import OpenAI
from typing import Union, Generator
from app.models.rag import RAGRequest, RAGResponse

# Tạo OpenAI client mới theo SDK >= 1.0.0
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_openai_from_rag(
    rag_request: RAGRequest, 
    rag_response: RAGResponse
) -> Union[str, Generator[str, None, None]]:
    messages = [
        {"role": "system", "content": "Bạn là một trợ lý hỗ trợ giáo dục thông minh."},
        {"role": "user", "content": rag_response.prompt}
    ]

    # Gọi ChatCompletion từ client mới
    response = openai_client.chat.completions.create(
        model=rag_request.model,
        messages=messages,
        stream=rag_request.stream
    )

    if rag_request.stream:
        def stream_generator():
            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        print(f"[Streaming chunk]: {delta.content}")  # 👈 In ra từng đoạn văn bản
                        yield delta.content
        return stream_generator()
    else:
        full_response = response.choices[0].message.content
        print(f"[Full response]: {full_response}")  # 👈 In ra toàn bộ câu trả lời nếu không stream
        return full_response

