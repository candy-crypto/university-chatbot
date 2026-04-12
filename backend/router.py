from fastapi import APIRouter
from pydantic import BaseModel

from retrieval import generate_grounded_answer
from db import log_chat


class ChatRequest(BaseModel):
    message: str
    department_id: str = "cs"

router = APIRouter()

'''
@router.get("/")
def root():
    return {"status": "ok", "message": "Backend is running"}

@router.get("/health")
def health():
    return {"status": "healthy"}

@router.post("/chat")
def chat(req: ChatRequest):
    return {
        "answer": f"Backend working. You selected {req.department_id}.",
        "department_id": req.department_id
    }
'''


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat")
async def chat(request: ChatRequest):
    result = generate_grounded_answer(request.message, request.department_id)

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "chunks": result.get("chunks", [])
    }
