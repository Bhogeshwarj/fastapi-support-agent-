"""Synthesize a cited answer from retrieved chunks.

Takes a question, retrieves + reranks relevant chunks, and asks the gateway
LLM to answer using only that context, citing sources by number.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from fastapi_support_agent.gateway import gateway_invoke
from fastapi_support_agent.rag.retrieval import build_reranked_retriever

SYSTEM_PROMPT = (
    "You are a support assistant for the FastAPI web framework. Answer the "
    "user's question using ONLY the numbered context chunks below - do not "
    "use outside knowledge. Cite every claim with the matching [n] marker. "
    "If the context doesn't contain the answer, say so plainly instead of "
    "guessing."
)


def _format_context(chunks) -> tuple[str, list[dict]]:
    context_lines = []
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        context_lines.append(f"[{i}] ({chunk.metadata['section']})\n{chunk.page_content}")
        citations.append(
            {
                "n": i,
                "url": chunk.metadata["url"],
                "section": chunk.metadata["section"],
                "source_file": chunk.metadata["source_file"],
            }
        )
    return "\n\n".join(context_lines), citations


def answer_question(question: str, top_n: int = 4) -> dict:
    retriever = build_reranked_retriever(top_n=top_n)
    chunks = retriever.invoke(question)
    context, citations = _format_context(chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n\n{context}\n\nQuestion: {question}"),
    ]
    response = gateway_invoke(messages)

    return {"answer": response.content, "citations": citations}


if __name__ == "__main__":
    result = answer_question("How do I add query parameters with validation?")
    print("Answer:", result["answer"])
    print()
    print("Citations:")
    for c in result["citations"]:
        print(f"  [{c['n']}] {c['url']} ({c['section']})")
