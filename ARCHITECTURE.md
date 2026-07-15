# Architecture (M1)

## System diagram: LangGraph shape, and where gateway/eval/guardrails sit

```mermaid
flowchart TD
    U[User query] --> IG[Input guardrail - M8]
    IG -->|blocked: off-topic / injection| REJ[Refusal response]
    IG -->|passed| P[Planner / router node - M6]

    P -->|single tool call needed| T{Tool selection}
    P -->|multi-part question| SA[Sub-agent delegation - M6]
    SA --> T

    T --> RAG[RAG retrieval tool - M3]
    T --> CL[Changelog / deprecation tool - M4]
    T --> IS[GitHub issue search tool - M4, live API]

    RAG --> AGG[Aggregate + synthesize with citations]
    CL --> AGG
    IS --> AGG

    AGG -->|claim needs approval| HITL[Human-in-the-loop checkpoint - M5]
    HITL -->|approved| OG
    AGG -->|no approval needed| OG[Output guardrail - M8]
    OG --> RESP[Final response]

    GW["LLM Gateway - M2 (provider fallback, cost tracking)"]
    GW -. wraps every model call .-> P
    GW -. wraps every model call .-> AGG
    GW -. wraps every model call .-> HITL

    EVAL["Eval harness - M7 (offline: golden set + LLM-as-judge)"]
    EVAL -. runs the whole graph, scores .-> RESP
```

**How to read this:**
- The graph itself (nodes/edges) is built in **M5** (basic router + tool-calling) and extended in **M6** (the sub-agent delegation branch, for questions that need more than one tool).
- The **gateway (M2)** isn't a node in the graph — it's a wrapper every node's LLM call goes through, which is exactly why we're building it *before* the graph exists: nothing needs retrofitting.
- **Guardrails (M8)** sit at the two edges of the graph — input (before the planner ever sees the query) and output (before the response leaves) — not scattered through the middle.
- **HITL (M5)** is one specific interrupt point: before the agent finalizes an answer that asserts something risky (a deprecation/breaking-change claim, or before it spends GitHub API quota on an ambiguous issue search). Not every response pauses for approval — just the ones matching that condition.
- **Eval (M7)** never runs inside this live request graph — it's a separate offline harness that replays the golden Q&A set through the same graph and scores the output afterward.

## Repo layout

```
fastapi-support-agent/
├── src/fastapi_support_agent/
│   ├── ingestion/     # pulls raw docs/changelog → data/raw/ (M1, this milestone)
│   ├── rag/           # chunking, embeddings, hybrid search (M3)
│   ├── gateway/        # provider fallback, cost tracking (M2)
│   ├── tools/           # changelog lookup, deprecation check, issue search (M4)
│   ├── agents/           # LangGraph graph, HITL, sub-agents (M5/M6)
│   ├── eval/              # golden set + judge (M7)
│   └── guardrails/         # M8
├── data/                    # gitignored, entirely regenerable via scripts/
├── scripts/                  # one-off ingestion/build CLI entrypoints
├── tests/
└── docker/                    # M9
```

## Data sources

| Source | What it's for | Acquisition | Freshness model |
|---|---|---|---|
| FastAPI docs (`docs/en/docs/**/*.md`) | RAG corpus for doc Q&A | Shallow + sparse `git clone` of `fastapi/fastapi@master`, `scripts/fetch_docs.py`, safe to re-run (`git pull` if already cloned) | Static snapshot, refreshed on demand |
| Changelog (`docs/en/docs/release-notes.md`) | Version lookup / "is X deprecated" tool (M4) | Comes free from the same sparse clone above — it lives inside `docs/en/docs/`, no separate fetch | Static snapshot, refreshed on demand |
| GitHub issues (`fastapi/fastapi` issue tracker) | Issue search tool (M4) — "has this been reported," known bugs, workarounds | Live query against the GitHub REST/Search API at ask-time, authenticated with a fine-grained PAT scoped to public-repo read-only (`GITHUB_TOKEN` in `.env`, see `.env.example`) | Always live — never bulk-downloaded, since freshness (open/closed state, new comments) matters more than a frozen copy |

Why issues aren't bulk-fetched like the docs: there are tens of thousands of them, and what makes them useful for support is current state, not a stale snapshot. Docs and the changelog are effectively immutable at a point in time, so a snapshot is fine there.

## Retrieval pipeline (M3)

- **Chunking** (`rag/chunking.py`) — two-stage split: `MarkdownHeaderTextSplitter` first (so every chunk knows its section), then `RecursiveCharacterTextSplitter` within each section (so chunk sizes are consistent, ~800 chars). Internal-only files (leading underscore, e.g. `_llm-test.md`) and `release-notes.md` are excluded — the changelog gets its own structured parser in M4. Each chunk carries `source_file`, `section`, and a `url` mapped back to the live `fastapi.tiangolo.com` site (not the GitHub repo). 1,595 chunks from the current corpus.
- **Embeddings** — local, via `langchain-huggingface`'s `HuggingFaceEmbeddings` running `sentence-transformers/all-MiniLM-L6-v2`. Chosen over an API-based embedding provider (e.g. Gemini's) specifically because indexing thousands of chunks would burn through a free-tier request quota fast — local embedding is unlimited and genuinely free forever.
- **Vector store** (`scripts/build_index.py`) — **Chroma**, persisted to `data/vector_store/chroma/` (gitignored, rebuilt from raw docs on demand). Chosen over FAISS (used previously in `RAG-Tutorials`) because every chunk's metadata travels with its embedding, which hybrid search and citations both depend on.
- **Hybrid search** (`rag/retrieval.py`, `build_hybrid_retriever`) — Chroma vector search + `BM25Retriever` (keyword matching, good at exact terms like class names) merged via `EnsembleRetriever`, which uses Reciprocal Rank Fusion. Returns the union of both retrievers' results, not trimmed to a fixed count.
- **Reranking** (`rag/retrieval.py`, `build_reranked_retriever`) — the merged hybrid candidates get rescored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, local, via `HuggingFaceCrossEncoder` + `CrossEncoderReranker`), then trimmed to a final `top_n`. Added after testing showed the raw hybrid merge included some weak keyword-only matches that a query+document-aware model correctly demotes.
- **Not yet built**: the synthesis step that takes reranked chunks + the user's question and calls `gateway_invoke()` to produce an actual cited answer. Retrieval returns chunks today, not answers.

Package note: in LangChain's v1 reorg, `EnsembleRetriever`, `ContextualCompressionRetriever`, and `CrossEncoderReranker` all moved out of the main `langchain` package into a separate `langchain-classic` package — found by testing imports directly rather than trusting older docs/tutorials.

## Dependency management

`uv`-managed project, Python 3.13. Every dependency added via `uv add` so versions are resolved live against PyPI and locked in `uv.lock` — no hand-typed version guesses. Current core deps: `langchain==1.3.13`, `langgraph==1.2.9`, `langsmith==0.10.3`, `chromadb==1.5.9`, `langchain-chroma==1.1.0`, `langchain-groq`, `langchain-google-genai`, `langchain-huggingface`, `sentence-transformers`, `langchain-community`, `rank-bm25`.

## Open decisions

- **Automated tests**: everything so far has been verified with manual smoke-test scripts (`scripts/test_gateway.py`, ad-hoc checks), not a `pytest` suite, even though `tests/` exists in the repo layout. M7's eval harness will be the main quality gate for RAG/agent behavior — not yet decided whether a lightweight `pytest` suite runs alongside it for the non-LLM plumbing (chunking, URL mapping, gateway fallback logic).
