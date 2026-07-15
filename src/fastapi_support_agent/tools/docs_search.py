"""RAG doc search exposed as an agent tool.

Unlike synthesis.answer_question() (a standalone direct-RAG entry point from
M3), this returns raw retrieved chunks with citation info as text - the
agent's own LLM turn does the synthesis, same as it does for the other three
tools. Keeps one consistent loop instead of nesting an LLM call inside a tool.
"""

from langchain_core.tools import tool

from fastapi_support_agent.rag.retrieval import build_reranked_retriever


@tool
def search_fastapi_docs(query: str) -> str:
    """Search FastAPI's official documentation for conceptual/how-to questions.

    Use this for "how do I..." or "what is..." questions about using FastAPI.
    Not for changelog/version/deprecation questions (use the changelog tools)
    or bug reports (use GitHub issue search).
    """
    retriever = build_reranked_retriever(top_n=4)
    chunks = retriever.invoke(query)
    if not chunks:
        return "No relevant documentation found."

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] ({chunk.metadata['section']}, {chunk.metadata['url']})\n"
            f"{chunk.page_content}"
        )
    return "\n\n".join(lines)
