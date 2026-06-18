"""The NovaStor customer assistant — a LangGraph ReAct agent, scoped per customer.

`build_assistant(customer_id, ...)` returns a compiled agent whose tools are
bound to that customer. The system prompt is the policy: ground knowledge
answers in the docs with citations, use account tools for account questions,
refuse out of scope, and never reveal other customers' data.
"""
from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.tools import build_tools
from app.data.operations import OperationsStore
from app.llm import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are NovaStor Logistics' assistant for a corporate customer. You help with "
    "two kinds of question and you must choose the right tool for each:\n\n"
    "1. NovaStor services, policies, SLAs, pricing, or contract terms -> call "
    "`search_knowledge_base` and answer using ONLY the returned passages, citing the "
    "[source] markers. If the passages do not contain the answer, say you don't know "
    "based on NovaStor's documentation. Never invent policy or pricing.\n"
    "2. The customer's own account, storage usage, contract dates, invoice, or "
    "shipments -> call `get_account_summary`, `get_shipment_status`, or "
    "`list_my_shipments`. Report only what the tools return.\n\n"
    "You only have access to THIS customer's account; you cannot see other customers' "
    "data, so never claim to. For anything unrelated to NovaStor (general trivia, etc.), "
    "politely decline. Be concise and professional."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def build_assistant(
    customer_id: str,
    retriever: VectorStoreRetriever,
    ops: OperationsStore,
    model: BaseChatModel | None = None,
):
    tools = build_tools(customer_id, retriever, ops)
    tools_by_name = {t.name: t for t in tools}
    if model is None:
        model = get_llm().bind_tools(tools)
    system = SystemMessage(content=SYSTEM_PROMPT)

    def agent_node(state: AgentState) -> dict:
        return {"messages": [model.invoke([system] + state["messages"])]}

    def tool_node(state: AgentState) -> dict:
        outputs = []
        for call in state["messages"][-1].tool_calls:
            t = tools_by_name.get(call["name"])
            if t is None:
                result = f"ERROR: unknown tool '{call['name']}'"
            else:
                try:
                    result = t.invoke(call["args"])
                except Exception as exc:  # a tool failure must not crash the loop
                    logger.exception("Tool %s failed", call["name"])
                    result = f"ERROR: {exc}"
            outputs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": outputs}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()
