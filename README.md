# University Chatbot

A full-stack New Mexico State University department chatbot that answers student questions with retrieval-augmented generation. The app combines crawled department web pages, NMSU catalog chunks, PostgreSQL course lookups, Weaviate hybrid search, and OpenAI answer generation.

The current project is centered on the NMSU Computer Science department (`cs`) and includes a Next.js chat UI, a FastAPI backend, web and catalog ingestion scripts, retrieval debugging output, and evaluation tooling.

## Current Features

- Next.js frontend with department selection, answer display, sources, retrieved chunks, and prompt context.
- FastAPI backend with `GET /health` and `POST /chat`.
- Unified Weaviate `DepartmentChunk` collection for both web and catalog content.
- Playwright web crawler for department pages configured by YAML.
- PDF catalog ingestion flow for 2025-2026 NMSU catalog chunks.
- PostgreSQL tables for crawl runs, chat logs, and course lookup records.
- Direct course lookup shortcuts for credits, General Education, and Viewing a Wider World questions.
- Redirect handling for real-time course availability questions that must be answered through Banner.
- Retrieval evaluation CSV export for every chat request.
- Evaluation harness with deterministic retrieval metrics and optional LLM-as-judge scoring.
- `MOCK_OPENAI=true` mode for testing the API and UI without OpenAI calls.

## Project Structure

```text
university-chatbot/
|- backend/
|  |- app.py                         # FastAPI app setup
|  |- router.py                      # API routes
|  |- retrieval.py                   # RAG, shortcuts, metadata boosts, prompt building
|  |- ingest.py                      # Department website crawler and web chunk ingestion
|  |- catalog_ingest.py              # Catalog PDF chunk ingestion
|  |- nmsu_catalog_chunker.py        # Catalog parsing pipeline
|  |- nmsu_course_chunker.py         # Course-specific catalog chunking helpers
|  |- evaluation_export.py           # Per-chat retrieval CSV export
|  |- db.py                          # PostgreSQL setup and lookup helpers
|  |- weaviate_client.py             # Local/cloud Weaviate connection and schema setup
|  |- requirements.txt
|  |- .env.example
|  |- configs/departments/cs.yaml
|  |- eval/                          # Evaluation harness, judge, ground truth, exports
|  `- evaluation/retrieval_eval.csv  # Generated chat evaluation log
|- frontend/
|  |- app/page.js                    # Chat UI
|  |- app/globals.css                # Chat UI styling
|  |- package.json
|  `- .env.local
|- scripts/                          # Project presentation helper scripts
|- output/                           # Generated presentation artifacts
|- docker-compose.yml
|- EVALUATION_PLAN.md
`- README.md
```

## How It Works

1. `backend/ingest.py` crawls configured department web pages and stores web chunks in Weaviate.
2. `backend/catalog_ingest.py` parses the NMSU catalog PDF, stores catalog chunks in Weaviate, and populates the PostgreSQL `courses` lookup table.
3. The frontend sends a question and department ID to `POST /chat`.
4. The backend first checks shortcut paths for current-semester availability, course credits, General Education, and Viewing a Wider World questions.
5. If no shortcut applies, the backend embeds the question, runs Weaviate hybrid search, applies local metadata boosts, builds grounded prompt context, and asks OpenAI for an answer.
6. The response includes the answer, sources, retrieved chunks, prompt context, and the path to the appended evaluation CSV.

## Tech Stack

- Frontend: Next.js 16, React 19, Tailwind CSS tooling
- Backend: FastAPI, Uvicorn, Pydantic
- AI: OpenAI Responses API and OpenAI embeddings
- Vector database: Weaviate
- Relational database: PostgreSQL
- Crawling: Playwright
- Catalog parsing: pdfplumber plus local chunking scripts
- Evaluation: CSV export, YAML ground truth, deterministic metrics, optional LLM judge

## Prerequisites

- Node.js 18+
- Python 3.11+
- Docker Desktop
- An OpenAI API key for real ingestion and real answers

For web ingestion, install Playwright browsers after installing backend dependencies:

```bash
python -m playwright install chromium
```

## Environment Setup

### Backend

Copy `backend/.env.example` to `backend/.env` and update the values.

```env
OPENAI_API_KEY=your_key_here
MOCK_OPENAI=false

OPENAI_CHAT_MODEL=gpt-5.5
JUDGE_MODEL=gpt-5.4-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
TOP_K=5
HYBRID_ALPHA=0.75

DATABASE_URL=postgresql://uniChatBotPostgres:uniChatBotPassword@localhost:5432/university_chatbot

WEAVIATE_MODE=local
WEAVIATE_HTTP_HOST=localhost
WEAVIATE_HTTP_PORT=8080
WEAVIATE_GRPC_HOST=localhost
WEAVIATE_GRPC_PORT=50051
WEAVIATE_COLLECTION=DepartmentChunk

CATALOG_PDF_PATH=../25-26 New Mexico State University - Las Cruces.pdf
CATALOG_YEAR=2025-2026
CATALOG_DEPARTMENT_ID=cs
CATALOG_SCRIPTS_DIR=.
```

Set `WEAVIATE_MODE=cloud` and provide `WEAVIATE_URL` plus `WEAVIATE_API_KEY` if using Weaviate Cloud.

### Frontend

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## Running Locally

### 1. Start Weaviate and PostgreSQL

From the project root:

```bash
docker compose up -d
```

This starts:

- Weaviate REST on `http://localhost:8080`
- Weaviate gRPC on `localhost:50051`
- PostgreSQL on `localhost:5432`

### 2. Install backend dependencies

From `backend/`:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium
```

### 3. Start the backend

From `backend/`:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health`
- `POST /chat`

Example `POST /chat` body:

```json
{
  "message": "Does CSCI 1115G count as general education?",
  "department_id": "cs"
}
```

Example response shape:

```json
{
  "answer": "...",
  "sources": ["..."],
  "chunks": [],
  "prompt_context": "...",
  "evaluation_csv": "backend/evaluation/retrieval_eval.csv"
}
```

### 4. Start the frontend

From `frontend/`:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Mock Mode

To test the API and frontend without using OpenAI tokens, set this in `backend/.env`:

```env
MOCK_OPENAI=true
```

In mock mode, `/chat` returns a deterministic answer, a mock source, a mock retrieved chunk, and prompt context. It skips OpenAI embedding and response-generation calls.

## Ingesting Content

The chatbot needs indexed content before real RAG answers work.

### Web Department Ingestion

From `backend/`:

```bash
python ingest.py
```

This uses `backend/configs/departments/cs.yaml`, crawls the configured CS department pages, embeds chunks, deletes previous `content_source=web` chunks for the department, and writes fresh web chunks into Weaviate.

### Catalog PDF Ingestion

From `backend/`:

```bash
python catalog_ingest.py
```

This parses the configured catalog PDF, maps catalog chunks into the unified Weaviate schema, deletes previous catalog chunks for the configured department and catalog year, inserts fresh catalog chunks, and upserts course records into PostgreSQL for direct course lookup.

Make sure `CATALOG_PDF_PATH` points to the local NMSU catalog PDF. The repository currently includes:

```text
25-26 New Mexico State University - Las Cruces.pdf
```

### Annual Re-ingestion

The catalog and department web pages change each academic year. To update the knowledge base:

1. Obtain the new NMSU Academic Catalog PDF and place it in the project root.
2. Update `CATALOG_PDF_PATH` and `CATALOG_YEAR` in `backend/.env` to point to the new PDF and year.
3. Re-run catalog ingestion: `python catalog_ingest.py` — this deletes previous catalog chunks for the department and year and inserts fresh ones.
4. Re-run web ingestion: `python ingest.py` — this deletes previous web chunks for the department and replaces them.
5. After re-ingestion, run the evaluation harness with `--update-notes` to refresh retrieval notes in `ground_truth.yaml` and verify that pass rates have not regressed.

Note that expected chunk IDs in `ground_truth.yaml` are derived from catalog page ranges and web URLs. If the new catalog reorganizes pages or the website restructures URLs, some expected chunk IDs may need to be updated before evaluation results are meaningful.

## Retrieval Behavior

`backend/retrieval.py` uses several layers:

- Direct Banner redirect for real-time availability, registration, seat, and current-semester offering questions.
- PostgreSQL course lookup for course credits, General Education suffixes, Viewing a Wider World suffixes, and lists of G/V courses.
- OpenAI query embeddings for semantic retrieval.
- Weaviate native hybrid search with BM25 plus vector ranking.
- Local metadata boosts for headings, course codes, degree metadata, chunk types, policy topics, course levels, and catalog authority.
- Grounded answer generation from retrieved context only.

The frontend intentionally displays retrieved chunks and prompt context to make retrieval debugging easier.

## Evaluation

Every `/chat` request appends rows to:

```text
backend/evaluation/retrieval_eval.csv
```

Each row captures the question, answer, sources, prompt context, retrieved chunk metadata, scores, and chunk text.

To run the evaluation harness:

```bash
cd backend
python eval/harness.py
```

Useful options:

```bash
python eval/harness.py --no-judge                        # skip LLM judge (faster, deterministic metrics only)
python eval/harness.py --category financial_aid          # run one category
python eval/harness.py --questions adv_001 adv_002       # run specific questions
python eval/harness.py --update-notes                    # refresh retrieval_note fields in ground_truth.yaml
```

Harness outputs are written to `backend/eval/results/` as JSONL result files, CSV score exports, and JSON summaries.

The evaluation question set and ground truth are defined in:

```text
backend/eval/ground_truth.yaml
```

Each entry specifies a question ID, category, question text, expected chunk IDs, expected source type, and key facts used for scoring. Edit this file to add, remove, or correct evaluation questions.

Additional export/import helpers live under `backend/eval/` for chunk IDs, collection exports, and department chunk snapshots.

## Database Tables

The backend initializes these PostgreSQL tables on startup:

- `courses`: catalog course lookup records used by direct course shortcuts.
- `chat_logs`: schema for chat history logging.
- `crawl_runs`: ingestion run metadata.

Current `/chat` responses are exported to the evaluation CSV. The route imports `log_chat`, but chat log insertion is not currently called.

## Department Configuration

Department crawling rules live under:

```text
backend/configs/departments/
```

The current checked-in department config is `cs.yaml`. It defines the department ID, root URL, seed URLs, allowed domains, allowed path prefixes, denied URL patterns, page types, levels, degree types, and campus metadata.

The frontend currently offers `cs`, `math`, and `unknown` buttons, but only `cs` has an ingestion config and indexed workflow in this repository.

## Useful Commands

```bash
# Start local infrastructure
docker compose up -d

# Run backend
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run web ingestion
cd backend
python ingest.py

# Run catalog ingestion
cd backend
python catalog_ingest.py

# Run retrieval evaluation without judge
cd backend
python eval/harness.py --no-judge

# Run frontend
cd frontend
npm run dev

# Build frontend
cd frontend
npm run build
```

## Current Limitations

- The main ingestion scripts are currently configured for the CS department.
- The frontend department picker includes options that do not yet have matching department configs.
- CORS is currently open to all origins in the backend.
- `chat_logs` exists, but the active `/chat` route writes evaluation CSV rows instead of inserting chat log records.
- Real-time registration and seat availability are intentionally redirected to Banner instead of answered from indexed content.
- Generated artifacts under `output/` and runtime logs may be local development artifacts rather than deployment inputs.

## License

Add a license before sharing or deploying this project publicly.
