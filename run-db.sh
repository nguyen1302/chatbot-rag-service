#!/bin/bash
echo "📦 Starting Qdrant database..."

DATA_DIR=$(pwd)/qdrant_data

if [ ! -d "$DATA_DIR" ]; then
  echo "📁 Creating $DATA_DIR..."
  mkdir -p "$DATA_DIR"
  chown 1000:1000 "$DATA_DIR"  # Qdrant chạy với UID 1000 trong container
fi

if docker ps -a --format '{{.Names}}' | grep -q '^qdrant$'; then
  echo "🔁 Restarting existing Qdrant container..."
  docker start qdrant
else
  echo "🚀 Running new Qdrant container..."
  docker run -d \
    --name qdrant \
    -p 6333:6333 \
    -v $DATA_DIR:/qdrant/storage \
    --restart unless-stopped \
    qdrant/qdrant
fi

echo "✅ Qdrant is running at http://localhost:6333"
