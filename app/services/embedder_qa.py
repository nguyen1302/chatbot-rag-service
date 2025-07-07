from pydantic import BaseModel
from typing import List
from openai import OpenAI
import os
from app.models.rag import RAGRequest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def extract_keywords(text: str, top_k: int = 5) -> str:
    tokens = word_tokenize(text, format="text").lower().split()
    stop_words = set(["là", "của", "và", "có", "cho", "này", "để", "khi", "vì", "những", "các", "tôi", "bạn", "rằng"])  # mở rộng nếu muốn
    filtered = [word for word in tokens if word not in stop_words and len(word) > 2]
    most_common = [word for word, _ in Counter(filtered).most_common(top_k)]
    return " ".join(most_common)

def process_rag_request(
    rag_request: RAGRequest,
    top_k: int = 3,
    sim_threshold: float = 0.47,
    delta_threshold: float = 0.05
) -> dict:
    messages = rag_request.messages

    # Bước 1: Lấy các cặp Q&A gần nhất (tối đa 5)
    qa_pairs = []
    for i in range(len(messages) - 1):
        if messages[i].role == "user" and messages[i + 1].role == "assistant":
            qa_text = f"Q: {messages[i].content} A: {messages[i + 1].content}"
            qa_pairs.append(qa_text)
    recent_qa = qa_pairs[-5:]

    # Bước 2: Lấy câu hỏi mới nhất của người dùng
    last_question = next((msg.content for msg in reversed(messages) if msg.role == "user"), "")
    if not last_question:
        return {
            "query": "",
            "query_keywords": "",
            "top_qas": [],
            "similarities": [],
            "key_q&a": [],
            "key_q_last": "",
            "is_followup": False,
            "followup_context": []
        }

    # Bước 3: Trích xuất từ khóa
    question_keywords = extract_keywords(last_question)
    qa_keywords_list = [extract_keywords(qa_text) for qa_text in recent_qa]

    if not qa_keywords_list:
        return {
            "query": last_question,
            "query_keywords": question_keywords,
            "top_qas": [],
            "similarities": [],
            "key_q&a": [],
            "key_q_last": question_keywords,
            "is_followup": False,
            "followup_context": []
        }

    # Bước 4: Embedding và tính tương đồng
    try:
        embedded_qa_keywords = [get_embedding(k) for k in qa_keywords_list]
        embedded_question_keywords = get_embedding(question_keywords)
        similarities = cosine_similarity(
            [embedded_question_keywords],
            embedded_qa_keywords
        )[0]
    except Exception as e:
        print("[Embedding or Similarity Error]", e)
        return {
            "query": last_question,
            "query_keywords": question_keywords,
            "top_qas": [],
            "similarities": [],
            "key_q&a": qa_keywords_list,
            "key_q_last": question_keywords,
            "is_followup": False,
            "followup_context": []
        }

    # Bước 5: Lọc các đoạn có độ tương đồng vượt ngưỡng
    selected_raw = [
        (i, similarities[i])
        for i in range(len(similarities))
        if similarities[i] > sim_threshold
    ]

    if not selected_raw:
        return {
            "query": last_question,
            "query_keywords": question_keywords,
            "top_qas": [],
            "similarities": [],
            "key_q&a": qa_keywords_list,
            "key_q_last": question_keywords,
            "is_followup": False,
            "followup_context": []
        }

    # Bước 6: Ưu tiên theo thứ tự thời gian hoặc độ tương đồng cao
    scores = [sim for _, sim in selected_raw]
    if max(scores) - min(scores) < delta_threshold:
        selected_sorted = sorted(selected_raw, key=lambda x: -x[0])  # mới nhất
    else:
        selected_sorted = sorted(selected_raw, key=lambda x: -x[1])  # giống nhất

    selected = selected_sorted[:top_k]
    relevant_qas = [recent_qa[i] for i, _ in selected]
    selected_similarities = [float(sim) for _, sim in selected]

    # Bước 7: Trả kết quả
    return {
        "query": last_question,
        "query_keywords": question_keywords,
        "top_qas": relevant_qas,
        "similarities": selected_similarities,
        "key_q&a": qa_keywords_list,
        "key_q_last": question_keywords,
        "is_followup": True,
        "followup_context": relevant_qas
    }
