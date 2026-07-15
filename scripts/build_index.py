"""Build the Chroma vector index from the chunked FastAPI docs.

One-time (or on-demand-refresh) build step - not part of the live query path.
Re-run this after re-running fetch_docs.py to pick up doc changes.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from fastapi_support_agent.rag.chunking import load_and_chunk_docs

PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store" / "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_index() -> None:
    print("Chunking docs...")
    chunks = load_and_chunk_docs()
    print(f"{len(chunks)} chunks to embed.")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    PERSIST_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Embedding + indexing into {PERSIST_DIR} (this may take a few minutes)...")
    start = time.time()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name="fastapi_docs",
    )
    print(f"Done in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    build_index()
