#!/bin/bash

echo "🔨 Building RAG backend..."

docker compose up --build -d rag-backend

echo "✅ RAG backend is running on http://localhost:8001"
