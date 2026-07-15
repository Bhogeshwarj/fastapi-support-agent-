"""Hybrid search: merges Chroma vector search with BM25 keyword search via
Reciprocal Rank Fusion (RRF, handled internally by EnsembleRetriever).
"""

from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from fastapi_support_agent.rag.chunking import load_and_chunk_docs

PERSIST_DIR = Path(__file__).resolve().parents[3] / "data" / "vector_store" / "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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


def build_reranked_retriever(top_n: int = 4, k: int = 5) -> ContextualCompressionRetriever:
    """Hybrid search, then rerank the merged candidates and keep only the top_n best."""
    base_retriever = build_hybrid_retriever(k=k)
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
