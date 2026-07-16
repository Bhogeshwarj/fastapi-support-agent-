# FastAPI Support/Ops Agent

A production-style support/ops agent for the FastAPI web framework — not a doc-QA demo.
It answers real support questions using hybrid-search RAG over the live docs, a structured
changelog parser for version/deprecation questions, live GitHub issue search, a multi-agent
LangGraph loop with a human-approval checkpoint, an eval harness, guardrails (including a
confirmed-and-fixed prompt injection vulnerability), and an LLM gateway with provider
fallback — deployed behind a real chat UI.

Built as a capstone project for an AI-engineering career transition — see
[`LEARNING_JOURNEY.md`](./LEARNING_JOURNEY.md) for a full concept-by-concept breakdown of
every decision and bug found along the way, and [`WRITEUP.md`](./WRITEUP.md) for the
short problem/approach/trade-offs/results summary.

## What it can do

- **Doc Q&A with real citations** — hybrid (vector + keyword) search, cross-encoder
  reranked, over FastAPI's actual documentation. Answers link back to real pages on
  `fastapi.tiangolo.com`.
- **"Is X deprecated?" / version lookup** — parses FastAPI's real changelog history
  (3,000+ entries) rather than guessing from an LLM's training data.
- **Live GitHub issue search** — checks whether a problem has already been reported,
  not a stale snapshot.
- **Multi-part questions** get decomposed and answered by isolated sub-agents, then
  combined into one answer.
- **Human-in-the-loop approval** — risky claims (deprecation/breaking-change assertions)
  pause for a human to approve or reject before the answer ships. Visible live in the chat UI.
- **Guardrails** — blocks off-topic/unsafe questions before they're processed, and a second
  check on every final answer before it leaves.
- **Free-tier LLM gateway** — Groq primary, Gemini automatic fallback, rate-limited to stay
  under free-tier caps, every call cost/latency-tracked.

## Architecture

```
question → input guardrail → planner → agent ⇄ tools  (or → sub-agents → aggregate)
                                            │
                                            ▼
                                     hitl_check → output guardrail → answer
```

Full diagram and the reasoning behind every node: [`ARCHITECTURE.md`](./ARCHITECTURE.md).

## Tech stack

Python 3.13, `uv`, LangChain / LangGraph 1.x, Groq + Google Gemini (via `langchain-groq` /
`langchain-google-genai`), Chroma + local `sentence-transformers` embeddings, BM25 hybrid
search, a local cross-encoder reranker, FastAPI + vanilla HTML/CSS/JS frontend, Docker,
Render.

## Eval results

Golden set of 8 real support questions, scored by an LLM-as-judge against pre-verified
reference answers (`scripts/run_eval.py`, results tracked in `eval_runs/` across commits):

| Metric | Score |
|---|---|
| Average correctness | 4.38 / 5 |
| Average citation accuracy | 4.25 / 5 |

Two of the eight are **deliberate known-failure cases**, included on purpose so these
numbers reflect real capability instead of only counting easy wins — see
`src/fastapi_support_agent/eval/golden_set.json` and the "Known limitations" section below.

## Running it

**Locally:**
```bash
uv sync
cp .env.example .env   # fill in GROQ_API_KEY, GOOGLE_API_KEY, GITHUB_TOKEN, LANGSMITH_API_KEY
uv run scripts/fetch_docs.py
uv run scripts/build_index.py
uv run uvicorn fastapi_support_agent.api.main:app --app-dir src --port 8000
```
Then open `http://localhost:8000`.

**Docker:**
```bash
docker build -f docker/Dockerfile -t fastapi-support-agent .
docker run -p 8000:8000 --env-file .env fastapi-support-agent
```
The doc corpus and vector index are fetched/built at container **startup** (not image build
time — Render's Docker build step can't reach `huggingface.co`, found via a real deploy
attempt), running in the background while the API starts serving immediately.

**Render:** connect this repo in the Render dashboard — it auto-detects `render.yaml` and
prompts for the four required secrets. Free tier spins down after 15 min idle (~1 min
cold-start on the next request, plus the background doc-fetch/index-build window).

**Eval:** `uv run scripts/run_eval.py`

## Known limitations

Tracked honestly rather than hidden — full detail in `ARCHITECTURE.md` and
`LEARNING_JOURNEY.md`:

- `check_deprecated` misses changelog entries older than FastAPI's PR-attribution
  convention (documented, included as an eval known-failure case).
- A sub-agent occasionally overrides correct tool output with its own outdated training
  knowledge when a GitHub issue search returns nothing — a grounding instruction reduces
  this but doesn't reliably eliminate it (confirmed unstable across repeated runs).
- Sub-agent LLM calls bypass the gateway's cost-tracking log (still get fallback/rate
  limiting, just not logged cost).
- No automated `pytest` suite — verification has been the eval harness plus deliberate
  manual/browser-driven testing at every step.
