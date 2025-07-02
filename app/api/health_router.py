from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["Health"])  # dùng full path
def health_check():
    return {"status": "ok"}
