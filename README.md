### RUN
docker compose up --build

### deploy 
docker compose up --build -d

### RUN local
uvicorn main:app --reload --port 8001