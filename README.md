pyenv local 3.10.14 # Áp dụng cho thư mục hiện tại
python --version    # -> 3.10.14
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### RUN
uvicorn main:app --reload --port 8001