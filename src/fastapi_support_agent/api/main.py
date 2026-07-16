"""HTTP API for the FastAPI support agent, plus the static chat UI.

Wraps build_agent_graph() (M5/M6) behind two endpoints that mirror LangGraph's
interrupt/resume pattern for the HITL checkpoint (M5): /chat runs the graph
and either returns a final answer or a "needs_approval" draft; /chat/resume
continues a paused thread with an approve/reject decision.
"""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

from fastapi_support_agent.agents.graph import build_agent_graph
from fastapi_support_agent.gateway.content import extract_text

app = FastAPI(title="FastAPI Support Agent")

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph


def kill_switch_active() -> bool:
    return os.environ.get("KILL_SWITCH_ENABLED", "false").lower() == "true"


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    decision: str


class ChatResponse(BaseModel):
    status: str  # "answered" | "needs_approval"
    thread_id: str
    answer: str | None = None
    draft_answer: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if kill_switch_active():
        raise HTTPException(status_code=503, detail="Service temporarily disabled.")

    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = get_graph().invoke({"messages": [("user", req.message)]}, config)

    if "__interrupt__" in result:
        draft = result["__interrupt__"][0].value.get("draft_answer", "")
        return ChatResponse(status="needs_approval", thread_id=thread_id, draft_answer=extract_text(draft))

    answer = extract_text(result["messages"][-1].content)
    return ChatResponse(status="answered", thread_id=thread_id, answer=answer)


@app.post("/chat/resume", response_model=ChatResponse)
def chat_resume(req: ResumeRequest):
    if kill_switch_active():
        raise HTTPException(status_code=503, detail="Service temporarily disabled.")

    config = {"configurable": {"thread_id": req.thread_id}}
    result = get_graph().invoke(Command(resume=req.decision), config)
    answer = extract_text(result["messages"][-1].content)
    return ChatResponse(status="answered", thread_id=req.thread_id, answer=answer)


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
