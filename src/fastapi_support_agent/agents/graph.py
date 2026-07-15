"""The core LangGraph agent: planning, sub-agent delegation, tool-calling,
and a human-in-the-loop checkpoint before risky claims.

Shape (see ARCHITECTURE.md for the full diagram):
    planner -> agent <-> tools -> hitl_check -> END          (simple questions)
    planner -> dispatch_subagents -> aggregate -> hitl_check -> END   (multi-part questions)

The planner decides which path a question needs. Simple questions go through
the plain M5 tool-calling loop unchanged. Questions with genuinely distinct
parts get decomposed and each part is dispatched to an isolated sub-agent
(fresh context, scoped to only the tools its domain needs) so the sub-tasks'
back-and-forth doesn't clutter one shared thread - then results are combined
in one aggregation step.
"""

from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from fastapi_support_agent.gateway.client import get_gateway_llm
from fastapi_support_agent.tools.changelog import check_deprecated, lookup_changelog_version
from fastapi_support_agent.tools.docs_search import search_fastapi_docs
from fastapi_support_agent.tools.github_issues import search_github_issues

TOOLS = [search_fastapi_docs, lookup_changelog_version, check_deprecated, search_github_issues]

SUBAGENT_TOOLS = {
    "docs": [search_fastapi_docs],
    "changelog": [lookup_changelog_version, check_deprecated],
    "issues": [search_github_issues],
}

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


class SubTask(BaseModel):
    subagent: Literal["docs", "changelog", "issues"] = Field(
        description="Which specialist handles this sub-task"
    )
    task: str = Field(description="The specific, self-contained question for that specialist")


class Plan(BaseModel):
    needs_delegation: bool = Field(
        description="True only if the question genuinely has multiple distinct parts "
        "requiring different kinds of lookups. Simple single-topic questions are False."
    )
    subtasks: list[SubTask] = Field(default_factory=list)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # Stored as a plain dict, not the Plan Pydantic model - LangGraph's
    # checkpointer warns that persisting arbitrary custom types will be
    # blocked in a future version unless explicitly registered.
    plan: dict | None
    subagent_results: list[str]


def build_agent_graph():
    llm = get_gateway_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    planner_llm = llm.with_structured_output(Plan)

    def planner_node(state: AgentState) -> dict:
        question = state["messages"][-1].content
        plan = planner_llm.invoke(
            [
                SystemMessage(
                    content="Decide if this FastAPI support question has multiple "
                    "distinct parts that each need a different kind of lookup (docs, "
                    "changelog, or GitHub issues)."
                ),
                HumanMessage(content=question),
            ]
        )
        return {"plan": plan.model_dump()}

    def route_after_planner(state: AgentState) -> str:
        plan = state.get("plan")
        if plan and plan.get("needs_delegation") and plan.get("subtasks"):
            return "dispatch_subagents"
        return "agent"

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

    def dispatch_subagents(state: AgentState) -> dict:
        subtasks = state["plan"]["subtasks"]
        results = []
        for subtask in subtasks:
            subagent = create_agent(model=llm, tools=SUBAGENT_TOOLS[subtask["subagent"]])
            sub_result = subagent.invoke({"messages": [HumanMessage(content=subtask["task"])]})
            final_message = sub_result["messages"][-1]
            results.append(f"[{subtask['subagent']}] {subtask['task']}\n-> {final_message.content}")
        return {"subagent_results": results}

    def aggregate_node(state: AgentState) -> dict:
        question = state["messages"][-1].content
        combined = "\n\n".join(state["subagent_results"])
        messages = [
            SystemMessage(
                content=SYSTEM_PROMPT + " Combine the sub-agent findings below into one "
                "coherent, cited answer to the original question."
            ),
            HumanMessage(content=f"Original question: {question}\n\nSub-agent findings:\n\n{combined}"),
        ]
        response = llm.invoke(messages)
        return {"messages": [response]}

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
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("dispatch_subagents", dispatch_subagents)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("hitl_check", hitl_check)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner", route_after_planner, {"dispatch_subagents": "dispatch_subagents", "agent": "agent"}
    )
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "hitl_check": "hitl_check"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("dispatch_subagents", "aggregate")
    graph.add_edge("aggregate", "hitl_check")
    graph.add_edge("hitl_check", END)

    return graph.compile(checkpointer=InMemorySaver())
