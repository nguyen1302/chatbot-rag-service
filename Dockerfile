FROM python:3.10-slim

# 1. Set timezone & install system deps (thêm libmupdf-tools cho PyMuPDF hoạt động ổn hơn)
ENV TZ=Asia/Ho_Chi_Minh
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Tạo venv riêng
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. Làm việc tại /app
WORKDIR /app

# 4. Copy deps trước để tối ưu caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ mã nguồn
COPY . .

# 6. Expose port + default CMD
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
