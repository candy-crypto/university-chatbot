# University Chatbot

A full-stack university department chatbot that crawls department websites, stores chunked content in Weaviate, and answers student questions with retrieval-augmented generation (RAG).

The project currently includes:

- A `Next.js` frontend for asking questions
- A `FastAPI` backend for chat requests
- A `Playwright`-based ingestion pipeline for crawling department pages
- `Weaviate` for vector storage and retrieval
- `PostgreSQL` for crawl and chat logging
- `OpenAI` for embeddings and grounded answer generation

## Project Structure

```text
university-chatbot/
|- backend/
|  |- app.py
|  |- router.py
|  |- retrieval.py
|  |- ingest.py
|  |- db.py
|  |- weaviate_client.py
|  |- requirements.txt
|  `- configs/departments/cs.yaml
|- frontend/
|  |- app/page.js
|  |- package.json
|  `- .env.local
|- docker-compose.yml
`- README.md
```

## How It Works

1. `backend/ingest.py` crawls a department website using Playwright.
2. Page text is cleaned, chunked, embedded with OpenAI, and stored in Weaviate.
3. The frontend sends a question to `POST /chat`.
4. The backend retrieves relevant chunks from Weaviate using hybrid ranking.
5. OpenAI generates an answer constrained to the retrieved context.

## Tech Stack

- Frontend: Next.js 16, React 19
- Backend: FastAPI, Uvicorn
- AI: OpenAI Responses API + OpenAI embeddings
- Vector DB: Weaviate
- Relational DB: PostgreSQL
- Crawling: Playwright
- Config: YAML + `.env`

## Prerequisites

- Node.js 18+
- Python 3.11+
- Docker Desktop
- An OpenAI API key

## Environment Setup

### 1. Backend environment

Copy `backend/.env.example` to `backend/.env` and fill in your values.

Example:

```env
OPENAI_API_KEY=your_key_here
OPENAI_CHAT_MODEL=gpt-5-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
TOP_K=5

DATABASE_URL=postgresql://uniChatBotPostgres:uniChatBotPassword@localhost:5432/university_chatbot

WEAVIATE_MODE=local
WEAVIATE_HTTP_HOST=localhost
WEAVIATE_HTTP_PORT=8080
WEAVIATE_GRPC_HOST=localhost
WEAVIATE_GRPC_PORT=50051
WEAVIATE_COLLECTION=DepartmentChunk
```

### 2. Frontend environment

Set `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## Running the Project

### 1. Start infrastructure

From the project root:

```bash
docker compose up -d
```

This starts:

- Weaviate on `http://localhost:8080`
- PostgreSQL on `localhost:5432`

### 2. Start the backend

From `backend/`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend endpoints:

- `GET /health` returns API status
- `POST /chat` accepts:

```json
{
  "message": "What courses are offered?",
  "department_id": "cs"
}
```

### 3. Start the frontend

From `frontend/`:

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Ingesting Department Content

Before the chatbot can answer grounded questions, crawl and index a department site:

```bash
cd backend
python ingest.py
```

The current ingestion entry point loads:

- `backend/configs/departments/cs.yaml`

That means the project is currently configured for the Computer Science department by default. The frontend shows `cs`, `math`, and `unknown`, but only `cs` has a department crawler config in the repository right now.

## Department Config

Department crawling rules live in YAML files under `backend/configs/departments/`.

The existing `cs.yaml` defines:

- Department ID
- Root URL
- Allowed domains
- Allowed path prefixes
- URL deny patterns

To support another department, add a new YAML config and update ingestion flow as needed.

## Database Notes

PostgreSQL tables are created automatically on backend startup:

- `chat_logs`
- `crawl_runs`

These are used to record chatbot interactions and ingestion runs.

## Retrieval Notes

The retrieval pipeline in `backend/retrieval.py` uses:

- OpenAI embeddings for semantic search
- Weaviate vector retrieval
- Local BM25-style lexical scoring
- Reciprocal rank fusion and metadata boosts

The final answer is generated only from retrieved context to reduce hallucinations.

## Current Limitations

- The default ingestion script is hardcoded to `configs/departments/cs.yaml`
- The frontend currently displays only the answer text and not returned sources
- Chat logging is implemented in `db.py`, but the active `/chat` route does not currently call `log_chat`
- CORS is currently open to all origins in the backend

## Future Improvements

- Add department configs for more programs
- Let users choose from dynamically loaded departments
- Display source URLs in the frontend
- Persist chat logs for every request
- Add tests for retrieval and ingestion
- Add authentication and admin tools for re-crawling content

## Useful Commands

```bash
# Start local services
docker compose up -d

# Run backend
cd backend
uvicorn app:app --reload

# Run ingestion
cd backend
python ingest.py

# Run frontend
cd frontend
npm run dev
```

## License

Add your preferred license here if this project will be shared or deployed publicly.
