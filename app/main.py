"""FastAPI application with all endpoints for the Sourdough Guru agent."""

import logging
import uuid
from pathlib import Path

# Configure logging so all sourdough.* loggers print to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("sourdough.main")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.models import ExecuteRequest, ExecuteResponse, StepTrace
from app.graph.workflow import sourdough_graph

app = FastAPI(title="Sourdough Guru", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
ARCHITECTURE_PNG = Path(__file__).resolve().parent.parent / "architecture.png"


# ---------- Endpoints ----------


@app.get("/api/team_info")
def team_info():
    return {
        "team_name": "Sourdough Guru",
        "students": [
            {"name": "Student 1", "id": "000000000"},
        ],
    }


@app.get("/api/agent_info")
def agent_info():
    return {
        "description": "Sourdough Guru is an AI-powered sourdough baking assistant that helps with factual Q&A, recipe recommendations, and bake-day planning.",
        "purpose": "Help bakers of all skill levels master sourdough through science-backed knowledge, customized recipes, and detailed bake schedules.",
        "prompt_templates": [
            "What temperature should I keep my starter at?",
            "Give me a recipe for a 75% hydration country loaf",
            "Plan my bake day — I need 20 sourdough loaves by 6am tomorrow",
        ],
        "examples": [
            {
                "prompt": "What is the ideal hydration for a beginner sourdough loaf?",
                "intent": "factual_qa",
                "summary": "Returns a grounded answer from sourdough books and research about optimal hydration levels for beginners.",
            },
            {
                "prompt": "Give me a recipe for sourdough focaccia with 80% hydration",
                "intent": "recipe",
                "summary": "Creates a detailed focaccia recipe with ingredients in grams, baker's percentages, and step-by-step method.",
            },
            {
                "prompt": "I need to have 10 loaves ready by 8am Saturday. Plan my bake.",
                "intent": "bake_plan",
                "summary": "Generates a backwards-calculated timeline with timestamps for each baking step.",
            },
        ],
    }


@app.get("/api/model_architecture")
def model_architecture():
    if ARCHITECTURE_PNG.exists():
        return FileResponse(ARCHITECTURE_PNG, media_type="image/png")
    raise HTTPException(status_code=404, detail="architecture.png not found")


@app.post("/api/execute")
async def execute(request: ExecuteRequest):
    session_id = request.session_id or str(uuid.uuid4())

    initial_state = {
        "messages": [],
        "user_query": request.prompt,
        "session_id": session_id,
        "intent": "",
        "intent_params": {},
        "retrieved_docs": [],
        "math_results": {},
        "bake_plan_data": {},
        "steps": [],
        "response": "",
    }

    try:
        result = sourdough_graph.invoke(initial_state)
    except Exception as e:
        logger.exception(f"[Execute] Graph error for query: {request.prompt}")
        return ExecuteResponse(
            status="error",
            error=str(e),
            response="Sorry, something went wrong processing your request.",
            steps=[],
        )

    steps = [
        StepTrace(
            module=s.get("module", ""),
            prompt=s.get("prompt"),
            response=s.get("response"),
        )
        for s in result.get("steps", [])
    ]

    return ExecuteResponse(
        status="ok",
        error=None,
        response=result.get("response", ""),
        steps=steps,
    )


@app.get("/api/sessions")
def list_sessions():
    """List all chat sessions for the sidebar."""
    from app.tools.bake_session import list_all_sessions
    try:
        return list_all_sessions()
    except Exception as e:
        logger.exception("[Sessions] Failed to list sessions")
        return []


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """Get messages for a specific session."""
    from app.tools.bake_session import load_session
    try:
        return load_session(session_id)
    except Exception as e:
        logger.exception(f"[Sessions] Failed to load session {session_id}")
        return []


@app.delete("/api/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    """Delete a chat session."""
    from app.tools.bake_session import delete_session
    try:
        delete_session(session_id)
        return {"status": "ok"}
    except Exception as e:
        logger.exception(f"[Sessions] Failed to delete session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bake_session/{session_id}/status")
def bake_session_status(session_id: str):
    from app.tools.bake_session import get_bake_status

    try:
        status = get_bake_status(session_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend
@app.get("/")
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"message": "Sourdough Guru API is running. Frontend not found."})
