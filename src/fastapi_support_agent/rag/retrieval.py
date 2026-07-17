"""Hybrid search: merges Chroma vector search with BM25 keyword search via
Reciprocal Rank Fusion (RRF, handled internally by EnsembleRetriever).

Embeddings go through Google's Gemini API rather than a local
sentence-transformers model: a local model drags in torch + transformers,
whose import footprint alone doesn't fit Render's 512MB free-tier container
(measured: a single process importing that stack sat at ~98% of the limit
before serving a single request). The API call trades a small amount of
per-query latency for cutting that dependency out entirely. Cross-encoder
reranking (previously HuggingFaceCrossEncoder) is dropped for the same
reason - RRF-merged hybrid ranking is used as-is, just trimmed to top_n.
"""

from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from fastapi_support_agent.rag.chunking import load_and_chunk_docs

PERSIST_DIR = Path(__file__).resolve().parents[3] / "data" / "vector_store" / "chroma"
EMBEDDING_MODEL = "gemini-embedding-001"


def build_hybrid_retriever(k: int = 5) -> EnsembleRetriever:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name="fastapi_docs",
    )
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})

    # BM25Retriever has no on-disk persistence - it's rebuilt in memory from
    # the same chunking function Chroma was built from, guaranteeing both
    # retrievers see identical chunks.
    chunks = load_and_chunk_docs()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])


def build_reranked_retriever(top_n: int = 4, k: int = 5) -> Runnable:
    """Hybrid search, trimmed to the top_n best RRF-ranked candidates."""
    base_retriever = build_hybrid_retriever(k=k)
    return base_retriever | RunnableLambda(lambda docs: docs[:top_n])
