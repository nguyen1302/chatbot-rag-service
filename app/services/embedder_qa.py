from pydantic import BaseModel
from typing import List
from openai import OpenAI
import os
from app.models.rag import RAGRequest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def process_rag_request(rag_request: RAGRequest, top_k: int = 3):
    messages = rag_request.messages

    # 1. Lấy 5 cặp Q&A gần nhất
    qa_pairs = []
    for i in range(len(messages) - 1):
        if messages[i].role == "user" and i + 1 < len(messages) and messages[i + 1].role == "assistant":
            pair = f"Q: {messages[i].content}\nA: {messages[i + 1].content}"
            qa_pairs.append(pair)

    # Giới hạn 5 cặp gần nhất
    recent_qa = qa_pairs[-5:]

    # 2. Lấy câu hỏi hiện tại (tin nhắn cuối từ người dùng)
    last_question = ""
    for msg in reversed(messages):
        if msg.role == "user":
            last_question = msg.content
            break

    # 3. Tính embedding cho các Q&A và câu hỏi cuối
    embedded_qa = [get_embedding(qa) for qa in recent_qa]
    embedded_question = get_embedding(last_question)

    # 4. Tính độ tương đồng cosine
    similarities = cosine_similarity([embedded_question], embedded_qa)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # 5. Trích các Q&A liên quan nhất
    relevant_qas = [recent_qa[i] for i in top_indices]

    return {
        "query": last_question,
        "top_qas": relevant_qas,
        "similarities": [float(similarities[i]) for i in top_indices]
    }
