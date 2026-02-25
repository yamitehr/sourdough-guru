"""Session management nodes: load/save chat history from/to Supabase."""

import logging

from langchain_core.messages import HumanMessage, AIMessage

from app.graph.state import SourdoughState
from app.tools.bake_session import save_session, load_session

logger = logging.getLogger("sourdough.session")


def load_session_node(state: SourdoughState) -> dict:
    """Load prior conversation messages from Supabase."""
    session_id = state.get("session_id", "")
    if not session_id:
        return {}

    try:
        stored_messages = load_session(session_id)
    except Exception as e:
        logger.warning(f"[LoadSession] Failed to load session {session_id}: {e}")
        return {}

    if not stored_messages:
        logger.info(f"[LoadSession] No prior messages for session {session_id}")
        return {}

    messages = []
    for msg in stored_messages:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    logger.info(f"[LoadSession] Loaded {len(messages)} prior messages for session {session_id}")
    return {"messages": messages}


def save_session_node(state: SourdoughState) -> dict:
    """Save the full conversation to Supabase."""
    session_id = state.get("session_id", "")
    if not session_id:
        return {}

    serializable = []
    for msg in state.get("messages", []):
        msg_type = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        if msg_type == "human":
            serializable.append({"role": "user", "content": content})
        elif msg_type == "ai":
            serializable.append({"role": "assistant", "content": content})

    serializable.append({"role": "user", "content": state.get("user_query", "")})
    if state.get("response"):
        serializable.append({"role": "assistant", "content": state["response"]})

    try:
        save_session(session_id, serializable)
        logger.info(f"[SaveSession] Saved {len(serializable)} messages for session {session_id}")
    except Exception as e:
        logger.warning(f"[SaveSession] Failed to save session {session_id}: {e}")

    return {}
