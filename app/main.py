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
        "group_batch_order_number": "1_10",
        "team_name": "Sourdough Guru",
        "students": [
            {"name": "Yamit Ehrlich", "email": "yamitehrlich@campus.technion.ac.il"},
            {"name": "Aviv Lugasi",   "email": "aviv.lugasi@campus.technion.ac.il"},
            {"name": "Sophia Danilov","email": "sophia.d@campus.technion.ac.il"},
        ],
    }


@app.get("/api/agent_info")
def agent_info():
    return {
        "description": "Sourdough Guru is an AI-powered sourdough baking assistant that helps with factual Q&A, recipe recommendations, and bake-day planning.",
        "purpose": "Help bakers of all skill levels master sourdough through science-backed knowledge, customized recipes, and detailed bake schedules.",
        "prompt_template": {
            "template": "Ask me anything about sourdough! Examples: 'What is the ideal fermentation temperature?', 'Give me a 75% hydration country loaf recipe', 'I need 8 loaves ready by 7am — plan my bake.'"
        },
        "prompt_examples": [
            {
                "prompt": "What is the ideal hydration for a beginner sourdough loaf?",
                "full_response": "For beginners, a hydration of 65–72% is ideal. Lower hydration doughs are easier to handle and shape since the gluten network is tighter, giving you more control. As you get comfortable with fermentation and shaping, you can gradually increase hydration to develop a more open crumb. Start at 68% and adjust from there.",
                "steps": [
                    {
                        "module": "supervisor",
                        "prompt": "What is the ideal hydration for a beginner sourdough loaf?",
                        "response": "{\"intent\": \"factual_qa\", \"intent_params\": {}}"
                    },
                    {
                        "module": "clarify",
                        "prompt": "What is the ideal hydration for a beginner sourdough loaf?",
                        "response": "{\"needs_clarification\": false}"
                    },
                    {
                        "module": "retriever",
                        "prompt": "ideal hydration beginner sourdough loaf",
                        "response": "Retrieved 4 relevant chunks from sourdough knowledge base."
                    },
                    {
                        "module": "factual_qa",
                        "prompt": "What is the ideal hydration for a beginner sourdough loaf?",
                        "response": "For beginners, a hydration of 65–72% is ideal..."
                    }
                ]
            },
            {
                "prompt": "Give me a recipe for sourdough focaccia with 80% hydration",
                "full_response": "**Sourdough Focaccia (80% Hydration)**\n\nIngredients (for 1 pan, ~600g dough):\n- Bread flour: 333g (100%)\n- Water: 267g (80%)\n- Sourdough starter (100% hydration): 67g (20%)\n- Salt: 7g (2%)\n- Olive oil: 20g (6%)\n\nMethod:\n1. Mix flour + water, autolyse 30 min.\n2. Add starter + salt, fold every 30 min for 2 hrs.\n3. Bulk ferment 4–6 hrs at 24°C.\n4. Transfer to oiled pan, dimple, rest 1 hr.\n5. Bake at 230°C for 22–25 min until golden.",
                "steps": [
                    {
                        "module": "supervisor",
                        "prompt": "Give me a recipe for sourdough focaccia with 80% hydration",
                        "response": "{\"intent\": \"recipe\", \"intent_params\": {\"target_product\": \"focaccia\", \"hydration\": 80}}"
                    },
                    {
                        "module": "clarify",
                        "prompt": "Give me a recipe for sourdough focaccia with 80% hydration",
                        "response": "{\"needs_clarification\": false}"
                    },
                    {
                        "module": "retriever",
                        "prompt": "sourdough focaccia recipe 80% hydration",
                        "response": "Retrieved 4 relevant chunks from sourdough knowledge base."
                    },
                    {
                        "module": "recipe",
                        "prompt": "Generate a full focaccia recipe with the computed quantities.",
                        "response": "**Sourdough Focaccia (80% Hydration)**..."
                    }
                ]
            }
        ],
    }


@app.get("/api/model_architecture")
def model_architecture():
    if ARCHITECTURE_PNG.exists():
        return FileResponse(ARCHITECTURE_PNG, media_type="image/png")
    raise HTTPException(status_code=404, detail="architecture.png not found")


@app.post("/api/execute")
async def execute(request: ExecuteRequest):
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")
    if len(prompt) > 5000:
        raise HTTPException(status_code=422, detail="Prompt exceeds 5000 character limit.")

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
            response=None,
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
