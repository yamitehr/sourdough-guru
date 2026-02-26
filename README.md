# Sourdough Guru

An AI-powered sourdough baking assistant built with **LangGraph**, **FastAPI**, and **RAG** (Retrieval-Augmented Generation). The agent answers sourdough questions grounded in real baking literature, generates customized recipes with baker's math, and builds detailed bake-day schedules with live progress tracking.

---

## Capabilities

### 1. Factual Q&A
Ask any question about sourdough science, techniques, troubleshooting, or ingredients. Answers are grounded exclusively in a knowledge base of 5 sourdough books and 44 research papers, with source citations.

> *"What temperature should I keep my starter at?"*

### 2. Recipe Recommendation
Get customized sourdough recipes with precise gram weights, baker's percentages, and step-by-step instructions. The agent asks clarifying questions (product type, hydration level) before generating.

> *"Give me a recipe for sourdough focaccia with 80% hydration"*

### 3. Bake-Day Planning
Generate a backwards-calculated baking timeline with concrete timestamps. The agent collects your constraints (deadline, number of loaves, kitchen temperature) and produces a detailed schedule. Active bake plans include in-app progress polling with browser notifications.

> *"Plan my bake day — I need 10 loaves ready by 6am tomorrow"*

---

## Architecture

```
User → FastAPI → LangGraph StateGraph → Response
```

### Graph Flow

```
load_session → supervisor → clarify ──→ (conditional routing)
                                          │
                    ┌─ needs_clarification → save_session → END
                    │
                    ├─ factual_qa → retrieve_context → generate_qa_answer → save_session → END
                    │
                    ├─ recipe → retrieve_context → compute_baking_math → generate_recipe → save_session → END
                    │
                    ├─ bake_plan → retrieve_context → build_timeline → generate_bake_plan → store_bake_session → save_session → END
                    │
                    └─ general → generate_general_response → save_session → END
```

### Key Design Decisions

- **Max 2 LLM calls per request**: supervisor (intent classification via structured output) + one generation node
- **Clarification node is pure Python** — checks for missing parameters per intent and asks the user before proceeding, no LLM cost
- **Baking math is deterministic** — hydration, baker's percentages, fermentation time estimation, and timeline scheduling are all pure Python calculations
- **RAG grounding** — all generation nodes are instructed to use ONLY retrieved context from the knowledge base and cite sources
- **Multi-turn memory** — conversation history is persisted to Supabase and loaded on each request

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| Agent orchestration | LangGraph (StateGraph) |
| LLM | LLMod.ai (OpenAI-compatible) — GPT-5-Mini Thinking |
| Embeddings | text-embedding-3-small via LLMod.ai |
| Vector store | Pinecone (cosine, 1536d) |
| Persistence | Supabase (PostgREST API via httpx) |
| Frontend | Single-page HTML/CSS/JS with marked.js |
| Deployment | Docker on Render |

---

## Project Structure

```
sourdough-guru/
├── app/
│   ├── main.py                     # FastAPI app + all endpoints
│   ├── config.py                   # Pydantic settings from env vars
│   ├── models.py                   # Request/response schemas
│   ├── graph/
│   │   ├── state.py                # LangGraph state definition
│   │   ├── workflow.py             # Graph assembly + compilation
│   │   └── nodes/
│   │       ├── supervisor.py       # Intent classification (structured output)
│   │       ├── clarify.py          # Parameter validation + clarifying questions
│   │       ├── retriever.py        # RAG retrieval from Pinecone
│   │       ├── factual_qa.py       # Grounded Q&A generation
│   │       ├── recipe.py           # Recipe generation + baking math
│   │       ├── bake_plan.py        # Timeline builder + bake plan generation
│   │       ├── general.py          # Greeting/chitchat handler
│   │       ├── session.py          # Load/save conversation history
│   │       ├── llm_utils.py        # Shared ChatOpenAI singleton
│   │       └── param_utils.py      # Safe parameter parsing
│   ├── tools/
│   │   ├── knowledge_base.py       # Pinecone vector search
│   │   ├── baking_math.py          # Deterministic baking calculations
│   │   └── bake_session.py         # Supabase CRUD (httpx)
│   └── ingestion/
│       ├── pdf_parser.py           # PDF text extraction (pdfplumber)
│       ├── chunker.py              # Text chunking (1000 chars, 200 overlap)
│       └── ingest.py               # CLI: parse → chunk → embed → Pinecone
├── frontend/
│   └── index.html                  # Chat UI (single page, no build tools)
├── dataset/
│   ├── Books/                      # Sourdough reference books (PDFs)
│   └── Research papers/            # Academic papers (PDFs)
├── requirements.txt
├── Dockerfile
├── render.yaml
└── .env.example
```

---

## Setup & Running

### Prerequisites

- Python 3.12+
- API keys for: LLMod.ai, Pinecone, Supabase

### 1. Clone and install

```bash
cd sourdough-guru
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 3. Create Supabase tables

Run this SQL in the Supabase SQL Editor:

```sql
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    messages JSONB DEFAULT '[]',
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bake_sessions (
    session_id TEXT PRIMARY KEY,
    plan_data JSONB DEFAULT '{}',
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### 4. Ingest knowledge base

Place PDFs in `dataset/Books/` and `dataset/Research papers/`, then run:

```bash
python -m app.ingestion.ingest
```

This parses all PDFs, chunks the text, generates embeddings via LLMod.ai, and upserts them into Pinecone. Only needs to be run once.

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/team_info` | Team name and student info |
| `GET` | `/api/agent_info` | Agent description, purpose, and examples |
| `GET` | `/api/model_architecture` | Architecture diagram (PNG) |
| `POST` | `/api/execute` | Main entry point — run the agent |
| `GET` | `/api/sessions` | List all chat sessions |
| `GET` | `/api/sessions/{id}/messages` | Get messages for a session |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `GET` | `/api/bake_session/{id}/status` | Poll bake plan progress |

### POST /api/execute

**Request:**
```json
{
    "prompt": "What temperature for my starter?",
    "session_id": "optional-uuid"
}
```

**Response:**
```json
{
    "status": "ok",
    "error": null,
    "response": "## Short Answer\n\nKeep your starter at **4-21°C (40-70°F)**...",
    "steps": [
        {"module": "Supervisor", "prompt": "...", "response": "..."},
        {"module": "KnowledgeBaseRetriever", "prompt": "...", "response": "..."},
        {"module": "FactualQAAgent", "prompt": "...", "response": "..."}
    ]
}
```

