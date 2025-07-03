#!/bin/bash

echo "🔨 Rebuilding RAG backend image (no cache)..."

docker compose build --no-cache rag-backend

echo "🚀 Restarting container..."
docker compose up -d rag-backend

echo "✅ RAG backend is running on http://localhost:8001"
