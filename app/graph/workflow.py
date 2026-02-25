"""LangGraph workflow assembly: wire all nodes into a StateGraph."""

from langgraph.graph import StateGraph, END

from app.graph.state import SourdoughState
from app.graph.nodes.supervisor import supervisor
from app.graph.nodes.clarify import clarify
from app.graph.nodes.retriever import retrieve_context
from app.graph.nodes.factual_qa import generate_qa_answer
from app.graph.nodes.recipe import compute_baking_math, generate_recipe
from app.graph.nodes.bake_plan import build_timeline, generate_bake_plan, store_bake_session
from app.graph.nodes.general import generate_general_response
from app.graph.nodes.session import load_session_node, save_session_node


def route_after_clarify(state: SourdoughState) -> str:
    """Route based on whether clarification was needed or we can proceed."""
    # If clarify set a response, we need to ask the user — skip to save_session
    if state.get("response"):
        return "save_session"

    intent = state.get("intent", "general")
    if intent == "factual_qa":
        return "retrieve_context_qa"
    elif intent == "recipe":
        return "retrieve_context_recipe"
    elif intent == "bake_plan":
        return "retrieve_context_bake"
    else:
        return "generate_general_response"


def build_graph() -> StateGraph:
    """Build and compile the Sourdough Guru workflow graph."""
    graph = StateGraph(SourdoughState)

    # Add all nodes
    graph.add_node("load_session", load_session_node)
    graph.add_node("supervisor", supervisor)
    graph.add_node("clarify", clarify)

    # Factual QA path
    graph.add_node("retrieve_context_qa", retrieve_context)
    graph.add_node("generate_qa_answer", generate_qa_answer)

    # Recipe path
    graph.add_node("retrieve_context_recipe", retrieve_context)
    graph.add_node("compute_baking_math", compute_baking_math)
    graph.add_node("generate_recipe", generate_recipe)

    # Bake plan path
    graph.add_node("retrieve_context_bake", retrieve_context)
    graph.add_node("build_timeline", build_timeline)
    graph.add_node("generate_bake_plan", generate_bake_plan)
    graph.add_node("store_bake_session", store_bake_session)

    # General path
    graph.add_node("generate_general_response", generate_general_response)

    # Session save
    graph.add_node("save_session", save_session_node)

    # Define edges
    graph.set_entry_point("load_session")
    graph.add_edge("load_session", "supervisor")
    graph.add_edge("supervisor", "clarify")

    # Conditional routing after clarify
    graph.add_conditional_edges(
        "clarify",
        route_after_clarify,
        {
            "save_session": "save_session",
            "retrieve_context_qa": "retrieve_context_qa",
            "retrieve_context_recipe": "retrieve_context_recipe",
            "retrieve_context_bake": "retrieve_context_bake",
            "generate_general_response": "generate_general_response",
        },
    )

    # Factual QA path edges
    graph.add_edge("retrieve_context_qa", "generate_qa_answer")
    graph.add_edge("generate_qa_answer", "save_session")

    # Recipe path edges
    graph.add_edge("retrieve_context_recipe", "compute_baking_math")
    graph.add_edge("compute_baking_math", "generate_recipe")
    graph.add_edge("generate_recipe", "save_session")

    # Bake plan path edges
    graph.add_edge("retrieve_context_bake", "build_timeline")
    graph.add_edge("build_timeline", "generate_bake_plan")
    graph.add_edge("generate_bake_plan", "store_bake_session")
    graph.add_edge("store_bake_session", "save_session")

    # General path edge
    graph.add_edge("generate_general_response", "save_session")

    # End
    graph.add_edge("save_session", END)

    return graph.compile()


# Compiled graph singleton
sourdough_graph = build_graph()
