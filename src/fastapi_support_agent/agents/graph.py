"""The core LangGraph agent loop: tool-calling with a human-in-the-loop
checkpoint before risky claims (deprecation/breaking-change assertions).

Shape (see ARCHITECTURE.md for the full diagram):
    agent <-> tools   (loop until the agent stops calling tools)
    agent -> hitl_check -> END   (once the agent has a final answer)
"""

from typing import Annotated

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from typing_extensions import TypedDict

from fastapi_support_agent.gateway.client import get_gateway_llm
from fastapi_support_agent.tools.changelog import check_deprecated, lookup_changelog_version
from fastapi_support_agent.tools.docs_search import search_fastapi_docs
from fastapi_support_agent.tools.github_issues import search_github_issues

TOOLS = [search_fastapi_docs, lookup_changelog_version, check_deprecated, search_github_issues]

SYSTEM_PROMPT = (
    "You are a support assistant for the FastAPI web framework. Use the "
    "available tools to answer questions accurately - search the docs for "
    "conceptual questions, the changelog tools for version/deprecation "
    "questions, and GitHub issue search for bug reports. Cite sources "
    "(doc URLs, PR links, issue links) in your final answer. If you don't "
    "have enough information after using the tools, say so plainly."
)

# Claims this risky get held for human approval before the answer is returned.
RISKY_KEYWORDS = ["deprecat", "no longer supported", "breaking change", "removed in"]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_agent_graph():
    llm_with_tools = get_gateway_llm().bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "hitl_check"

    def hitl_check(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        content = str(last_message.content)
        if not any(kw in content.lower() for kw in RISKY_KEYWORDS):
            return {}

        decision = interrupt(
            {
                "reason": "This answer asserts a deprecation/breaking-change claim - "
                "review before it's sent.",
                "draft_answer": content,
            }
        )
        if decision != "approve":
            return {"messages": [AIMessage(content=str(decision))]}
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("hitl_check", hitl_check)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "hitl_check": "hitl_check"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("hitl_check", END)

    return graph.compile(checkpointer=InMemorySaver())
