from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from router import router
from db import init_db

app = FastAPI(title="University Chatbot Backend")


origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

class ChatRequest(BaseModel):
    message: str
    department_id: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    try:
        init_db()
    except Exception as e:
        print(f"DB init skipped or failed: {e}")

'''
@app.post("/chat")
def chat(req: ChatRequest):
    return {
        "answer": f"Chatbot received: {req.message}",
        "department_id": req.department_id,
    }
'''

app.include_router(router)

'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    department_id: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    return {
        "answer": f"Chatbot received: {req.message}",
        "department_id": req.department_id,
    }
'''