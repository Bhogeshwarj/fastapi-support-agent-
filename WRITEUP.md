# Write-up: FastAPI Support/Ops Agent

## Problem

Most "AI support agent" portfolio projects are a single RAG pipeline over documentation —
they answer "how do I do X," but not "is X still supported," "has anyone else hit this
bug," or "what actually changed between these two versions." Real support work needs all
three, plus enough safety and reliability engineering that the thing doesn't confidently
say something false or leak internal instructions. I built this as a capstone project for
an AI-engineering career transition specifically to demonstrate that broader, more
production-shaped scope — not just "I can build a chatbot over some PDFs."

## Approach

The system is a LangGraph agent, not a single RAG chain: an input guardrail filters
off-topic/unsafe questions before anything else runs; a planner decides whether a question
is simple (answered by one ReAct-style tool-calling loop) or genuinely multi-part (in which
case it's decomposed and each part is answered by an isolated sub-agent with its own clean
context, then combined); a human-in-the-loop checkpoint pauses before any answer that
asserts a deprecation or breaking-change claim; and an output guardrail does a final check
before anything ships. Every LLM call in every one of those nodes goes through a single
gateway that picks between two free-tier providers (Groq primary, Gemini fallback), retries
on failure, self-throttles to stay under rate limits, and logs cost.

The RAG layer itself is hybrid search (vector + BM25) merged by reciprocal rank fusion,
then reranked by a cross-encoder before synthesis — chosen after empirically comparing
results, not by default. The changelog/deprecation tooling deliberately isn't RAG at all:
it's a regex parser over the real changelog, because "is X deprecated" needs an exhaustive,
exact answer, not a top-k similarity guess. Quality is measured by an eval harness — a
golden set of real questions with pre-verified reference answers, scored by an LLM-as-judge,
with results tracked across git commits so regressions are visible over time.

## Trade-offs

- **Free-tier providers over paid ones** — genuinely free forever (no billing risk), at the
  cost of needing careful rate-limiting and provider fallback, which became a real feature
  rather than an afterthought.
- **Baking the vector index into the deploy vs. relying on a persistent disk** — Render's
  free tier isn't guaranteed a persistent disk, so the docs/index are rebuilt at container
  startup instead, trading a slower cold start for a fully stateless, self-contained
  container.
- **Prompt-based grounding over a heavier architectural fix** — a sub-agent's tendency to
  override tool output with its own pretrained assumptions (see Results) was reduced with
  a system-prompt instruction rather than a more complex programmatic contradiction-detector.
  This was a deliberate scope call: document the residual instability honestly rather than
  over-engineer a fix for a project at this scale.
- **No automated test suite** — verification leaned on the eval harness plus deliberate,
  repeated manual and browser-driven testing at every step, rather than `pytest`. A
  reasonable choice for a project this size, a real gap at larger scale.

## Results

- **4.38/5 average correctness, 4.25/5 average citation accuracy** across an 8-question
  golden set — including two questions deliberately kept as known-failure cases so the
  score reflects real capability, not just easy wins.
- **Two real bugs found and fixed via actual red-teaming and browser testing, not
  code review alone**: a confirmed prompt-injection vulnerability (a crafted tool result
  got the agent to relay dangerous security advice as legitimate, until a grounding
  instruction closed it — verified across repeated runs) and a UI bug where Gemini's
  structured response format leaked as a raw Python repr into the human-approval screen.
- **A real deployment failure found and fixed independently**: Render's Docker build step
  can't reach `huggingface.co`, breaking the embedding model download at build time — fixed
  by moving that step to container startup, running in the background while the API starts
  serving immediately.
- **One documented, unresolved limitation, tracked rather than hidden**: a sub-agent
  sometimes overrides correct tool output with its own outdated training belief when a
  search legitimately returns nothing — reduced, not eliminated, and said so plainly in the
  eval set and docs.
