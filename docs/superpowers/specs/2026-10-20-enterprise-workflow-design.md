# KnowHub enterprise workflow — approved scope

The user approved an internal knowledge retrieval and ticket handling engine, followed by a verified AI resume update and GitHub delivery. Existing DeepSeek reuse was selected. This implements that scope, not the superseded GraphRAG proposal.

## Product and architecture
An operator asks about a ticket; the system retrieves SOPs, reads current ticket/order data through typed tools, explains the evidence, and stages a resolution. Explicit confirmation is required before a transaction updates the ticket. Rejection leaves business state unchanged; duplicate approval creates no duplicate audit.

FastAPI serves a React/TypeScript workspace. PostgreSQL stores documents, 384-dimensional multilingual-e5 embeddings via pgvector, synthetic business records, runs and audit. BM25 and dense candidates use reciprocal rank fusion (not a cross-encoder reranker). LangGraph runs a bounded DeepSeek native Function Calling loop. MCP exposes the same read tools only. Operational traces exclude hidden model reasoning.

## Boundaries and acceptance
Single authorized workspace, bearer API token, localhost default. PDF/Markdown/text ingestion preserves source/page/section. No real customer data, production tenancy, GraphRAG, model training, Redis, K8s or cloud deployment. Legacy sources/data preserved. User-owned credentials are read by path, never copied; existing Kage teacher budgets/jobs untouched.

Acceptance requires real PostgreSQL/pgvector, semantic embeddings and DeepSeek: retrieval, missing-ID clarification, order lookup, stage/approve/reject, persisted runs and audit. Verify desktop/mobile UI, authorization, idempotency and failure behavior. Publish raw RAG-only versus tool-enabled task outputs with methodology; no fabricated or broad benchmark claims. Resume lists only shipped behavior.

## Model budget
Per run: at most 4 model calls, 12 tool calls and 1200 completion tokens/call. Initial validation cap: 40 live model calls including connection probe; broader evaluation is a later decision. Nothing in this scope extends historical Kage experiment budgets.
