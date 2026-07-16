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

# Bake the doc corpus + vector index into the image at build time - Render's
# free tier isn't guaranteed a persistent disk, so a self-contained image
# (rebuilt from public FastAPI docs, no secrets needed here) is safer across
# the free tier's spin-down/spin-up cycles than relying on a mounted volume.
RUN uv run scripts/fetch_docs.py && uv run scripts/build_index.py

# Render sets $PORT at runtime (defaults to 10000) - must not hardcode it.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn fastapi_support_agent.api.main:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
