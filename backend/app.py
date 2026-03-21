# app.py
# pip install fastapi uvicorn python-dotenv pyyaml

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your Vercel frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    department_id: str | None = None

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat")
def chat(req: ChatRequest):
    return {
        "answer": f"Backend working. You selected {req.department_id}.",
        "department_id": req.department_id
    }