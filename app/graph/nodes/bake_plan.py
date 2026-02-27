"""Bake planning node: build timeline and generate natural language plan."""

import json
import logging
from datetime import datetime, timedelta

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.graph.state import SourdoughState
from app.graph.nodes.llm_utils import get_llm
from app.graph.nodes.param_utils import safe_float, safe_int
from app.tools.baking_math import (
    estimate_fermentation_time,
    default_bread_steps,
    calculate_timeline,
)

logger = logging.getLogger("sourdough.bake_plan")

SYSTEM_PROMPT = """You are the Sourdough Guru, a bake-day planning expert.

CRITICAL: Use ONLY the provided baking timeline data and context documents. Do NOT add steps, temperatures, or techniques from your general training knowledge. The timeline was computed from verified baking science — present it faithfully.

Given the baking timeline with concrete timestamps, create a clear, friendly bake plan:
1. Summarize the overall schedule (start time -> finish time)
2. Walk through each step with its timestamp and what to do
3. For sensory cues and tips, ONLY use information from the provided context documents and cite the source
4. Highlight critical timing points (when the baker MUST take action)
5. Note any steps where the baker can rest/sleep

Formatting:
- Use **Markdown** formatting for readability
- Use ## headings for major sections (## Overview, ## Timeline, ## Tips)
- Use **bold** for all timestamps and critical action items
- Use a table for the timeline overview (| Time | Step | Duration |)
- Mark rest/sleep periods with a relaxed tone (e.g., "You can sleep now!")
- Use > blockquotes for tips from sources
- End with a **Sources** section listing the documents referenced"""


def build_timeline(state: SourdoughState) -> dict:
    """Build a deterministic baking timeline from parameters."""
    params = state.get("intent_params", {})

    temp_c = safe_float(params.get("temperature_c"), 24)
    hydration = safe_float(params.get("hydration"), 75)
    starter_pct = safe_float(params.get("starter_pct"), 20)
    num_loaves = safe_int(params.get("num_loaves"), 1)

    bulk_hours = estimate_fermentation_time(temp_c, hydration, starter_pct)
    steps = default_bread_steps(bulk_hours, num_loaves)

    # start_time and ready_by are alternative anchors — start_time takes priority
    constraints = {}
    start_time = None
    if params.get("start_time"):
        start_time = datetime.fromisoformat(params["start_time"])
    elif params.get("ready_by"):
        constraints["ready_by"] = params["ready_by"]

    timeline = calculate_timeline(steps, start_time=start_time, constraints=constraints)

    logger.info(f"[Timeline] Built {len(timeline)} steps, bulk={bulk_hours}h, temp={temp_c}C, loaves={num_loaves}")

    bake_plan_data = {
        "timeline": timeline,
        "num_loaves": num_loaves,
        "bulk_fermentation_hours": bulk_hours,
        "temperature_c": temp_c,
        "hydration": hydration,
    }

    # Check if the calculated start time is in the past
    if timeline:
        now = datetime.now()
        plan_start = datetime.fromisoformat(timeline[0]["start_time"])
        if plan_start < now:
            total_minutes = sum(s["duration_minutes"] for s in steps)
            earliest_finish = now + timedelta(minutes=total_minutes)
            bake_plan_data["infeasible"] = True
            bake_plan_data["infeasible_details"] = {
                "requested_ready_by": params.get("ready_by", ""),
                "required_start": plan_start.strftime("%Y-%m-%d at %H:%M"),
                "now": now.strftime("%Y-%m-%d at %H:%M"),
                "total_hours": round(total_minutes / 60, 1),
                "earliest_finish": earliest_finish.strftime("%Y-%m-%d at %H:%M"),
                "earliest_finish_iso": earliest_finish.isoformat(),
            }
            logger.warning(
                f"[Timeline] Infeasible: plan would start {plan_start.strftime('%H:%M')} "
                f"but it's already {now.strftime('%H:%M')}"
            )

    return {"bake_plan_data": bake_plan_data}


def generate_bake_plan(state: SourdoughState) -> dict:
    """Generate a natural-language bake plan from the timeline."""
    plan_data = state.get("bake_plan_data", {})

    # Infeasibility check — return early without calling the LLM
    if plan_data.get("infeasible"):
        d = plan_data["infeasible_details"]
        answer = (
            f"**That deadline isn't reachable — the bake would need to have started at "
            f"{d['required_start']}, which has already passed.**\n\n"
            f"Here's why: your bake requires **{d['total_hours']} hours** from start to finish "
            f"(including bulk fermentation and overnight cold retard). "
            f"Working backwards from your requested ready-by time puts the start in the past.\n\n"
            f"**What you can do:**\n"
            f"- **Start right now** — if you begin immediately, your loaves will be ready by "
            f"**{d['earliest_finish']}**.\n"
            f"- **Pick a later deadline** — tell me a new ready-by time and I'll build a fresh plan.\n\n"
            f"Would you like me to plan for **{d['earliest_finish']}** or a different time?"
        )
        logger.info(f"[BakePlan] Returned infeasibility notice (start was {d['required_start']})")
        step = {
            "module": "bake_plan_infeasible",
            "prompt": state["user_query"],
            "response": answer,
        }
        return {"response": answer, "steps": [step]}

    llm = get_llm()

    context_parts = []
    for doc in state.get("retrieved_docs", [])[:3]:
        context_parts.append(f"[{doc.get('source', '?')}]: {doc.get('text', '')}")
    context = "\n\n".join(context_parts)

    timeline = plan_data.get("timeline", [])

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in state.get("messages", [])[-6:]:
        role = getattr(msg, "type", "user")
        content = getattr(msg, "content", str(msg))
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))

    user_prompt = f"""Context from knowledge base:
{context}

Baking timeline:
{json.dumps(timeline, indent=2)}

Additional info:
- Bulk fermentation: {plan_data.get('bulk_fermentation_hours', '?')} hours
- Number of loaves: {plan_data.get('num_loaves', 1)}
- Kitchen temperature: {plan_data.get('temperature_c', '?')}C

User request: {state['user_query']}

Create a detailed, friendly bake plan:"""

    messages.append(HumanMessage(content=user_prompt))

    logger.info(f"[BakePlan] Generating plan for: {state['user_query']}")

    response = llm.invoke(messages)
    answer = response.content.strip()

    logger.info(f"[BakePlan] Response length: {len(answer)} chars")

    if not answer:
        logger.warning(f"[BakePlan] Empty answer!")

    step = {
        "module": "bake_plan",
        "prompt": user_prompt,
        "response": answer,
    }

    return {
        "response": answer,
        "steps": [step],
    }


def store_bake_session(state: SourdoughState) -> dict:
    """Save the bake plan to Supabase for polling notifications."""
    from app.tools.bake_session import save_bake_plan

    plan_data = state.get("bake_plan_data", {})
    session_id = state.get("session_id", "")

    if plan_data.get("infeasible"):
        logger.info(f"[StoreBakeSession] Skipping save — plan is infeasible")
        return {}

    if plan_data and session_id:
        try:
            save_bake_plan(session_id, plan_data)
            logger.info(f"[StoreBakeSession] Saved plan for session {session_id}")
        except Exception as e:
            logger.warning(f"[StoreBakeSession] Failed to save: {e}")

    return {}
