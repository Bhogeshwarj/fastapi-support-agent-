# git is needed for scripts/fetch_docs.py's sparse clone of fastapi/fastapi.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY scripts/ scripts/

# Doc corpus + vector index are built at container startup, not image build
# time - confirmed via a real Render deploy that Render's Docker BUILD step
# can't reach huggingface.co (embedding model download failed there), while
# the running container clearly has full internet access (GitHub issue
# search and the Groq/Gemini gateway both depend on it at runtime). Render's
# free tier isn't guaranteed a persistent disk anyway, so rebuilding from the
# public FastAPI docs on every container start (no secrets needed) is
# consistent with the original no-disk-dependency design, just moved later.

# Render sets $PORT at runtime (defaults to 10000) - must not hardcode it.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uv run scripts/fetch_docs.py && uv run scripts/build_index.py && uv run uvicorn fastapi_support_agent.api.main:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
