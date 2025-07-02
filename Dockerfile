FROM python:3.10-slim

# Set timezone & install system deps
ENV TZ=Asia/Ho_Chi_Minh
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual env (tốt hơn cài trực tiếp vào global site-packages)
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working dir
WORKDIR /app

# Copy deps & install sớm (caching tốt hơn)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Port & startup
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
