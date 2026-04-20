from fastapi import APIRouter
from pydantic import BaseModel

from evaluation_export import append_chat_evaluation_row
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
    export_path = append_chat_evaluation_row(
        question=request.message,
        department_id=request.department_id,
        result=result,
    )

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "chunks": result.get("chunks", []),
        "prompt_context": result.get("prompt_context", ""),
        "evaluation_csv": export_path,
    }
