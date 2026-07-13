# FastAPI Support/Ops Agent

A support/ops agent for the FastAPI open-source project — combining a documentation Q&A layer
(hybrid RAG with citations) with real support tooling (changelog/version lookup, deprecation
checks, GitHub issue search), a LangGraph multi-agent loop with human-in-the-loop approval,
deep-agent-style sub-task delegation, an eval harness with LLM-as-judge scoring, guardrails
against off-topic queries and prompt injection, and an LLM gateway layer with provider
fallback and cost tracking.

Built as a capstone project for an AI engineering career transition, fusing Phases 1-8 of a
personal roadmap (LangChain fundamentals, RAG, eval/guardrails, gateways, LangGraph agents,
deep agents, production, portfolio) into one end-to-end system.

Status: early scaffolding — architecture and milestone plan in progress (M1).
