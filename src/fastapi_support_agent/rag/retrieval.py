"""Hybrid search: merges Chroma vector search with BM25 keyword search via
Reciprocal Rank Fusion (RRF, handled internally by EnsembleRetriever).
"""

from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from fastapi_support_agent.rag.chunking import load_and_chunk_docs

PERSIST_DIR = Path(__file__).resolve().parents[3] / "data" / "vector_store" / "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_hybrid_retriever(k: int = 5) -> EnsembleRetriever:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
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
