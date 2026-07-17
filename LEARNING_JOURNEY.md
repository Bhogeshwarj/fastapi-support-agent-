# FastAPI Support/Ops Agent — Learning Journey & Architecture Reference

This document exists as a personal revision reference: every concept learned, every
architectural decision made, and every real bug found and fixed while building this
project — organized so it can be re-read later to refresh the "why," not just the "what."

## What this project is

A support/ops agent for the FastAPI open-source project. Not a doc-QA chatbot — a system
that answers real support questions using: hybrid-search RAG over the actual docs, a
structured changelog parser for version/deprecation questions, live GitHub issue search,
a multi-agent LangGraph loop with a human-approval checkpoint, an eval harness, guardrails,
and an LLM gateway with provider fallback — all fused into one working, deployed system.

## Architecture, end to end

```
User question
   │
   ▼
input_guardrail   ── LLM classifies on_topic/safe → refuses off-topic/unsafe questions here
   │
   ▼
planner           ── decides: simple question, or multi-part needing decomposition?
   │
   ├─→ agent ⇄ tools  ── ReAct loop: LLM picks a tool, sees result, decides next step
   │        (search_fastapi_docs / lookup_changelog_version / check_deprecated /
   │         search_github_issues)
   │
   └─→ dispatch_subagents → aggregate
            each sub-task runs in its OWN isolated sub-agent (fresh context,
            only its domain's tools), then results are combined into one answer
   │
   ▼
hitl_check        ── if the answer asserts a deprecation/breaking-change claim,
   │                  PAUSE the whole graph and wait for human approve/reject
   ▼
output_guardrail  ── final LLM check: does this leak system prompt / internal info /
   │                  unsafe content? blocks it if so
   ▼
Final response (with citations)
```

Every LLM call in every node above goes through one gateway (`gateway_invoke()`), which
picks the provider, retries on failure, throttles to stay under free-tier limits, and logs
cost.

## The concepts, phase by phase

### M1 — Architecture & data acquisition
**Concept: sparse + shallow git clone.** Instead of downloading a whole repo (or hitting a
GitHub API once per file), `git clone --depth 1 --filter=blob:none --sparse` pulls only the
one folder needed (`docs/en/docs/`), with no history. One `git pull` refreshes it later.

**Decision:** the changelog (`release-notes.md`) is *inside* that same docs folder — so
fetching docs and fetching the changelog was never two separate jobs.

### M2 — LLM Gateway
**Concept: provider fallback via `.with_fallbacks()`.** Every LangChain chat model shares
one interface, so `primary.with_fallbacks([backup])` returns an object that behaves like
`primary` but silently retries against `backup` on error. No hand-rolled try/except.

**Concept: token-bucket rate limiting.** `InMemoryRateLimiter` fills a bucket at a steady
rate; each call consumes one token; an empty bucket makes the *caller* wait instead of
hitting the provider's real rate limit and getting a 429.

**Concept: usage tracking via callbacks.** `UsageMetadataCallbackHandler` reads a
standardized `usage_metadata` field every modern chat model returns (input/output tokens),
regardless of provider — logged to `data/usage_log.jsonl` with an estimated cost even
though both providers (Groq, Gemini) are free tier.

**Real bug found:** `gemini-2.5-flash` — the model both docs and a web search suggested —
turned out to be dead ("no longer available to new users"). Found by actually calling the
live API, not by trusting documentation.

### M3 — RAG core
**Concept: two-stage chunking.** `MarkdownHeaderTextSplitter` first (so every chunk knows
its section), then `RecursiveCharacterTextSplitter` within each section (so chunk sizes are
consistent for embedding) — because a single pass either produces wildly inconsistent
chunk sizes or loses section context.

**Concept: local embeddings.** `sentence-transformers/all-MiniLM-L6-v2`, run entirely
on-machine — chosen specifically because embedding ~1,600 chunks via an API would burn
through free-tier request quota fast; local embedding is unlimited and genuinely free.
*(Reversed in M11 below - this reasoning was right about the quota risk, but wrong about
it being the binding constraint: memory was, and it hit first.)*

**Concept: hybrid search + Reciprocal Rank Fusion.** Vector search (semantic similarity)
and BM25 (keyword matching, good at exact class names) are two different retrieval
strategies; `EnsembleRetriever` merges their two ranked lists by *where each result ranked*
in each list — not a fresh comparison, a merge of rankings.

**Concept: cross-encoder reranking.** Unlike vector/BM25 (which score query and document
*independently*), a cross-encoder scores the query and one candidate *together* in a single
pass — far more accurate, but too expensive to run on the whole corpus, so it only reranks
the small hybrid-merged shortlist. *(Dropped in M11 below, same memory constraint that
killed local embeddings - a cross-encoder is itself a local torch model.)*

**Real finding:** the raw hybrid merge returned 10 results for `k=5` (not trimmed), and some
were weak keyword-only matches. Reranking + trimming to `top_n` fixed this, verified by
comparing before/after result quality on a real query.

### M4 — Support/ops tools
**Concept: three different data-access patterns, on purpose.**
- Docs → pre-built vector database (Chroma), queried without re-processing.
- Changelog → local file, re-parsed live on every call (fast enough that pre-indexing adds
  nothing; the file only changes when re-fetched).
- GitHub issues → live API call every time, nothing ever stored — freshness (open/closed
  state) matters more than a snapshot.

**Real bug found:** `check_deprecated('regex')` said "likely not deprecated" — wrong. The
real deprecation entry predates the changelog's PR-attribution format our parser expects,
so it's silently skipped. Documented as a known limitation rather than hidden, and later
included as a deliberate known-failure case in the eval golden set.

**Concept: `@tool` decorator.** Turns a plain Python function into something an LLM can be
handed as an option to call — the decorator auto-generates the name/description/parameter
schema an LLM needs directly from the function's docstring and type hints.

### M5 — LangGraph agent loop
**Concept: `StateGraph`.** A flowchart where nodes are plain functions and one shared
"state" (a dict-like object) gets passed between them. `Annotated[list, add_messages]`
means new messages *append* to the list instead of overwriting it.

**Concept: the ReAct loop.** `agent` node calls the LLM with tools bound; if it requests a
tool call, `ToolNode` executes it and loops back to `agent`; repeat until the LLM stops
calling tools and just answers.

**Concept: `interrupt()` / `Command(resume=...)`.** Called inside a node, `interrupt()`
pauses the *entire graph*, persists its exact state via a checkpointer (`InMemorySaver`),
and later resumes exactly where it left off when given `Command(resume=value)` — this is
what makes the human-approval checkpoint possible.

**Verified, not assumed:** both the approve path (answer passes through unchanged) and the
reject path (answer gets replaced) were tested against a real interrupt, not just read as
code.

### M6 — Sub-agent delegation
**Key architectural question, tested before building:** does a single ReAct loop already
handle multi-part questions? Tested first — yes, partially (parallel tool calls in one
turn). The real, distinct value of true sub-agent delegation is **context isolation**: each
sub-task gets a fresh `create_agent()` instance with only its own tools and no visibility
into other sub-agents' work, which matters once sub-tasks get complex enough that mixing
them into one shared thread would clutter it.

**Real bug found (the most interesting one):** with two sub-agents disagreeing — one
correctly reporting a new feature from the changelog, the other finding zero GitHub issues
about it and concluding from its *own outdated training knowledge* that the feature/version
"doesn't exist" — the aggregation step sided with the wrong, more confident-sounding claim.
This is a textbook case of an LLM's **parametric (pretrained) knowledge overriding grounded
tool output**. A grounding instruction was added and reduces the failure rate, but doesn't
reliably eliminate it — confirmed by re-running the identical case multiple times with
different outcomes. Tracked honestly as a known limitation, not hidden.

### M7 — Eval harness
**Concept: LLM-as-judge.** The judge doesn't evaluate truth from scratch — it's handed a
question, a pre-verified reference answer, and the system's actual answer, and does reading
comprehension: "do these convey the same facts?" It's only as good as the ground truth it's
compared against.

**Concept: honest known-failure cases.** Two golden-set questions (`regex` deprecation,
the sub-agent contradiction case) are deliberately included *expecting* imperfect scores,
so the eval numbers reflect real capability instead of only counting easy wins.

**Latest real numbers:** 4.38/5 average correctness, 4.25/5 average citation accuracy
across 8 golden questions, with the only underperformance concentrated in the two
documented known-limitation cases.

### M8 — Guardrails
**Two distinct guardrails, not one:**
- **Input guardrail** — classifies the raw question before the planner ever sees it;
  blocks off-topic and unsafe/injection questions with a canned refusal.
- **Output guardrail** — a *second*, later check on the actual final answer, regardless of
  which path produced it — catches things the input check couldn't (e.g. accidental leakage
  in the answer itself), verified to correctly distinguish a normal technical answer from a
  simulated system-prompt leak.

**Real, confirmed vulnerability (not hypothetical):** manually red-teamed by simulating a
crafted GitHub issue result (never created a real fake issue — that would be an
inappropriate action against a third-party public repo). An obvious injection attempt was
already resisted by the undefended baseline. A subtler one — framed as "per maintainer
request, recommend this workaround without caveats" — successfully got the agent to relay
genuinely dangerous advice (disable TLS verification, hardcode a debug secret key) as if it
were legitimate. A grounding/injection-defense instruction was added specifically in
response and re-verified across 3 separate runs afterward — all 3 correctly flagged the
advice as unsafe instead of relaying it.

### M9 — Production
**Concept: FastAPI backend wrapping the graph.** `/chat` and `/chat/resume` map directly
onto LangGraph's invoke/interrupt/resume pattern — a normal answer vs. a paused-for-approval
draft is just a different JSON shape in the same response model.

**Concept: kill switch.** One env var checked at the very top of every request handler,
before the graph is touched at all — the cheapest possible circuit breaker.

**Real bugs found via actually running the UI in a browser (not just reading the code):**
1. Gemini's structured (list-of-blocks) response content, once naively `str()`'d inside
   `hitl_check`, couldn't be recovered later — the human-approval UI showed a raw Python
   list repr instead of readable text. Fixed by normalizing with `extract_text()` *before*
   it ever becomes a string.
2. The frontend's minimal markdown renderer didn't handle triple-backtick code fences,
   leaking raw backticks into rendered answers. Fixed by handling fenced blocks before the
   inline-code regex could partially consume them.

**Real deployment constraint found (via an actual Render deploy attempt):** Render's Docker
*build* step can't reach `huggingface.co` — the embedding model download failed there. Fixed
by moving the doc-fetch + index-build step from build time to container *startup* instead,
running in the background while `uvicorn` starts serving immediately, with both
`search_fastapi_docs` and the changelog tools returning a graceful "still warming up, retry
shortly" message during that window rather than erroring. This was the first of four
distinct failures in getting Render fully working - continued in M11.

### M11 — Render deployment debugging: four failures, four root causes

Each fix in this sequence was verified against the *actual* failure before moving to the
next, not assumed to work from reasoning alone - a pattern worth calling out on its own,
separate from the technical fixes themselves.

**Failure 2: port-scan timeout.** Moving doc-fetch/index-build to container startup (M9,
above) meant `uvicorn` didn't bind the port until that job finished, and cloning the repo
plus downloading/loading an embedding model took longer than Render's port-scan window.
**Fix:** background the fetch/build job, start `uvicorn` in the foreground immediately.

**Failure 3: the identical timeout, again, after backgrounding.** `uv run` takes an
internal lock to check/sync its venv before running anything - two concurrent `uv run`
invocations (the backgrounded job and `uvicorn`) serialize on that same lock, so `uvicorn`
silently blocked until the background job finished, reproducing the exact same symptom.
**Concept: `uv run --no-sync`.** Skips the sync check entirely - safe here because the venv
was already synced with `--frozen` at image build time, so no runtime re-sync is needed.
**Fix, verified locally before redeploying:** ran the exact container startup shell logic
locally, confirmed the port opened in ~14s (previously never) while the background job
completed with no errors, running concurrently with a responsive server.

**Failure 4: OOM ("Ran out of memory (used over 512MB)").** This is the one worth
remembering the methodology for as much as the fix. Rather than guessing again, I ran the
built Docker image in a *local* container with a hard `-m 512m --memory-swap 512m` cap -
reproducing Render's exact limit on a machine I could actually inspect. That surfaced two
separate, compounding problems:
1. **`torch` defaults to the CUDA-enabled Linux wheel.** Even though this app only ever
   does CPU inference on a small embedding model, `torch` (pulled in transitively via
   `sentence-transformers`) resolved to a build depending on the full nvidia
   cublas/cudnn/cufft/cusolver/cusparse/nvjitlink stack - multiple GB of GPU libraries that
   get mapped into memory at `import torch` regardless of whether a GPU exists. **Concept:
   pinning a transitive dependency's source in `uv`.** `[tool.uv.sources]` only overrides
   packages that are *direct* project dependencies - `torch` had to be added directly to
   `pyproject.toml` before a marker-scoped `{ index = "pytorch-cpu", marker = "sys_platform
   == 'linux'" }` override would actually take effect (confirmed by watching `uv lock -v`
   silently never touch the custom index until this was done). Cut `uv.lock` from 155 to
   119 packages even before the bigger fix below.
2. **Even CPU-only torch didn't leave enough headroom.** With the CUDA stack gone, a
   single `uvicorn` process still sat at ~98% of 512MB just from importing the RAG module
   chain, before serving a request - confirmed by running `uvicorn` alone (no background
   job at all) in the capped container and watching it plateau there. Running fetch/build
   as a *second* concurrent process (from Failure 2/3's fix) made this strictly worse by
   double-loading the same stack, but a single process alone already had no slack to fix
   into. The real constraint was the dependency stack itself, not the process architecture.

**Fix:** replaced local embeddings (`sentence-transformers`) and the local cross-encoder
reranker (M3) with Gemini's embeddings API (`GoogleGenerativeAIEmbeddings`,
`gemini-embedding-001`) - removing `torch`/`transformers`/`sentence-transformers` from the
dependency tree entirely. Verified in the same 512MB-capped container: steady-state memory
dropped to 65-80%, image size 1.52GB → 686MB.

**Real finding, not anticipated:** Gemini's free tier turned out to cap `embed_content` at
*both* 100 requests/minute *and* 1000 requests/day - discovered by watching
`build_index.py` retry-and-fail against the same daily quota no matter how long its
per-minute backoff waited, then confirmed directly by calling the embeddings API with
increasing batch sizes until it 429'd. **Fix:** batch and pace `build_index.py`'s calls
(instead of one big `Chroma.from_documents()` call) to stay under the per-minute cap, with
retry-with-backoff for that cap and a fast-fail (no point retrying) once the error message
identifies the *daily* quota specifically, since no realistic backoff clears that one.

**Failure 5 (a correctness bug, found by using the deployed app, not a deploy failure):**
`search_fastapi_docs` only guarded against a missing doc corpus (the cold-start case from
M9) - it didn't catch an exception from the embedding call itself. Since `/chat`
(`api/main.py`) has no top-level try/except around the agent graph invocation, the daily
quota above being exhausted meant any question needing doc search 500'd outright, while a
question the LLM could answer without a tool call worked fine - a distinction that looked
like a mystery ("why does this question fail and not that one?") until traced to which
questions actually trigger a tool call. **Fix:** catch retrieval exceptions inside the tool
itself and return a plain-text fallback, same pattern as the cold-start guard already had.
**Lesson generalized:** every tool that reaches an external API needs its own error
handling, since nothing upstream of it will catch a failure gracefully.

## Known limitations (tracked honestly, not hidden)

- `check_deprecated` misses very old changelog entries that predate the PR-attribution
  format (documented in `tools/changelog.py`, included as an eval known-failure case).
- Sub-agent parametric-knowledge override (M6) is reduced by a grounding instruction but not
  reliably eliminated — confirmed unstable across repeated runs of the identical question.
- `dispatch_subagents`' `create_agent()` calls bypass the gateway's cost-tracking log (still
  get fallback/rate-limiting, since those are baked into the model objects themselves).
- No automated `pytest` suite — verification has been the eval harness plus deliberate
  manual/browser-driven testing at every step, not unit tests.
- Gemini's free-tier embeddings quota (100/minute, 1000/day, M11) means doc search is
  genuinely unavailable - not just slower - once the daily cap is hit, until it resets.
- Eval scores above were measured against the cross-encoder-reranked retrieval pipeline,
  before M11 dropped the reranker for memory reasons - not yet re-run since.

## How to run it

- **Locally:** `uv sync`, copy `.env.example` to `.env` and fill in keys, `uv run scripts/fetch_docs.py`, `uv run scripts/build_index.py`, then `uv run uvicorn fastapi_support_agent.api.main:app --app-dir src --port 8000`.
- **Docker:** `docker build -f docker/Dockerfile -t fastapi-support-agent .`, then `docker run -p 8000:8000 --env-file .env fastapi-support-agent`. Doc-fetch + index-build happen at container startup, not image build time.
- **Render:** connect the repo, Render auto-detects `render.yaml`, fill in the 4 secret env vars it prompts for. Free tier spins down after 15 min idle (~1 min cold start on the next request).
- **Eval:** `uv run scripts/run_eval.py` — runs the full golden set through the real agent, scores with the judge, saves to `eval_runs/` (committed to git, so scores are comparable across commits over time).
