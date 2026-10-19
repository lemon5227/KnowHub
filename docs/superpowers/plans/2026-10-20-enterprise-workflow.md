# Enterprise workflow Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for scoped work and independent review. User authorized continuous execution, GitHub push and AI resume update.

**Goal:** Real knowledge retrieval and ticket execution with native tool calls, explicit confirmation and evidence-bound resume delivery.

**Architecture:** FastAPI + LangGraph connects the model to typed read/search/stage tools. PostgreSQL/pgvector persists knowledge, synthetic business records, runs and transactions. React supplies the operator UI.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, psycopg, pgvector, multilingual-e5-small, BM25/RRF, DeepSeek, React/TypeScript/Vite, MCP, Docker.

## Global Constraints
- Work on codex/enterprise-workflow in fresh /Users/wenbo/Projects/KnowHub; preserve legacy sources and local data.
- No secret copying/output; localhost binding; one authorized workspace.
- Initial live checks: max40 model calls, max1200 completion tokens/call. Per run max4 model calls and12 tool calls.
- All public data synthetic; no GraphRAG or production use claim.

### Task 1: Durable store and retrieval
Files: knowhub/store.py, knowhub/retrieval.py, knowhub/seed.py, tests/test_store.py, tests/test_retrieval.py, examples/knowledge/*.md.
Interfaces: Store(url).initialize(), put_document(), save_run(), approve_run(run_id, approved); Retriever(store, embedder).ingest(name,text,source), search(query,k,mode). JSON-serializable results.
- [ ] Failing tests: no write before approval; repeat approval creates one audit; rejected/stale proposals do not write; absent methods produce RED.
- [ ] Run uv run pytest tests/test_store.py -q against isolated knowhub_test; implement transactions/validation and run GREEN.
- [ ] Retrieval tests: distinct lexical/semantic candidates, preserved metadata, duplicate upload replaces chunks, blank query yields no unrelated results. Observe RED, implement dense+BM25+RRF, GREEN; use real embeddings separately.

### Task 2: Agent and API
Files: knowhub/model.py, knowhub/agent.py, knowhub/api.py, knowhub/mcp_server.py, tests/test_agent.py, tests/test_api.py.
Interfaces: Model.complete(messages,tools)->dict; Agent.run(question,ticket_id=None,mode='agent')->run dict; create_app(store,agent,token)->FastAPI.
- [ ] Failing tests: malformed/unknown tool args, tool result propagation, exhausted budget, model failure, forged citations, missing API authorization, no writes from model tool calls.
- [ ] Implement DeepSeek Function Calling and LangGraph decision/tools loop; persist outputs, explicit failures and public traces; GREEN.
- [ ] API contract in docs/api-contract.md: public health only; bearer-protected tickets/documents/uploads/runs/decisions. MCP read tools share implementations.
- [ ] Validate actual PostgreSQL/embeddings/DeepSeek; never count fixture responses as live acceptance.

### Task 3: Operator workspace
Files: frontend/package.json, frontend/src/App.tsx, frontend/src/styles.css, frontend/vite.config.ts, frontend/tsconfig.json.
Contracts: docs/api-contract.md; bearer token in sessionStorage only.
- [ ] Responsive UI: login, inbox, composer, knowledge upload/list, answer/sources/trace, approval/rejection, errors, persisted run history. No fake metrics or answers.
- [ ] Strict TypeScript/build; actual browser smoke verifies API results and approved writes, desktop/mobile layout.

### Task 4: Validation and delivery
Files: README.md, scripts/evaluate.py, artifacts/evaluation/*, Dockerfile, .github/workflows/ci.yml; existing AI resume source/PDF outside repo.
- [ ] Hand-authored RAG-only versus tool-enabled cases requiring SOP, live business data and clarification. Save original outputs, usage and elapsed times.
- [ ] Whole-branch review; fix material issues; full tests/build/live UI acceptance.
- [ ] Resume:3–4 verified KnowHub bullets, GitHub link, 2025 prototype/2026 upgrade; preserve unrelated facts. Compile PDF, extract text, render and inspect.
- [ ] Secret/data audit, commit and push codex branch; create attached PR without merging main. Retain old implementation in Git history and on disk.
