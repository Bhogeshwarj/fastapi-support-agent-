"""Build the Chroma vector index from the chunked FastAPI docs.

One-time (or on-demand-refresh) build step - not part of the live query path.
Re-run this after re-running fetch_docs.py to pick up doc changes.
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from fastapi_support_agent.rag.chunking import load_and_chunk_docs

PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store" / "chroma"
EMBEDDING_MODEL = "gemini-embedding-001"

# Gemini's free tier caps embed_content at 100 requests/minute. Chroma sends
# one request per ~100-doc batch, and firing every batch back-to-back blows
# through that limit on a corpus this size (confirmed via a real 429:
# RESOURCE_EXHAUSTED). Batch small and pace requests instead of one big
# Chroma.from_documents() call, with backoff as a safety net for whatever
# per-minute quota is left on the key at build time.
#
# There's also a separate free-tier cap of 1000 embed_content requests/DAY
# (quotaId "EmbedContentRequestsPerDayPerProjectPerModel-FreeTier") - that
# one won't clear in the ~10 minutes a per-minute backoff loop can cover, so
# it's detected separately below and fails fast instead of retrying for
# nothing. A build that fails this way just needs to wait for the daily
# quota to reset and re-run (or retry itself on the next container start).
BATCH_SIZE = 50
PACE_SECONDS = 5
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"](\d+)")


def _add_with_retry(vector_db: Chroma, batch: list, attempt: int = 1) -> None:
    try:
        vector_db.add_documents(batch)
    except Exception as e:
        if "PerDay" in str(e):
            print("Daily embedding quota exhausted - not retrying until it resets.")
            raise
        if attempt >= 6 or "RESOURCE_EXHAUSTED" not in str(e):
            raise
        match = _RETRY_DELAY_RE.search(str(e))
        delay = int(match.group(1)) + 5 if match else 30 * attempt
        print(f"Rate limited, retrying in {delay}s (attempt {attempt})...")
        time.sleep(delay)
        _add_with_retry(vector_db, batch, attempt=attempt + 1)


def build_index() -> None:
    print("Chunking docs...")
    chunks = load_and_chunk_docs()
    print(f"{len(chunks)} chunks to embed.")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    PERSIST_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Embedding + indexing into {PERSIST_DIR} (this may take a few minutes)...")
    start = time.time()
    vector_db = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name="fastapi_docs",
    )
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        _add_with_retry(vector_db, batch)
        print(f"Indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks...")
        if i + BATCH_SIZE < len(chunks):
            time.sleep(PACE_SECONDS)
    print(f"Done in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    build_index()
