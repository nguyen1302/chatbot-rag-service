import os
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_question(question: str) -> List[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=question
    )
    embedding = response.data[0].embedding
    print(f"[DEBUG] Embedding length: {len(embedding)}")
    if embedding is None:
        raise ValueError("❌ Failed to get embedding from OpenAI.")
    return embedding