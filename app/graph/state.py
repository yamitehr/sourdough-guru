"""LangGraph state definition for the Sourdough Guru agent."""

import operator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class SourdoughState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    user_query: str
    session_id: str
    intent: str           # "factual_qa" | "recipe" | "bake_plan" | "general"
    intent_params: dict   # extracted parameters (e.g., target product, constraints)
    retrieved_docs: list  # RAG results
    math_results: dict    # baking math output (optional)
    bake_plan_data: dict  # structured timeline (optional)
    steps: Annotated[list[dict], operator.add]  # accumulates across nodes
    response: str         # final answer
