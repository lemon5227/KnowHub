# Enterprise evaluation implementation plan

**Goal:** Publish reproducible, honestly bounded retrieval and real-model workflow measurements.
**Architecture:** Fixed JSON data -> isolated PostgreSQL and existing engine -> raw JSON -> deterministic aggregate + manual review -> report/resume.
**Tech Stack:** Existing Python, psycopg, pgvector, E5, DeepSeek, pytest, LaTeX.
**Constraints:** See ../specs/2026-10-23-benchmark-design.md. Work in the existing clean feature/enterprise-workflow branch; do not reset demo data, alter retrieval, or create AI authorship.

- [ ] Freeze `benchmarks/enterprise-v1/{corpus,queries,workflows}.json` and SHA256 manifest before running. Section IDs are gold units; include multi-source questions.
- [ ] Write failing tests in `tests/test_benchmark.py` for partial multi-evidence recall, duplicate hits, missing evidence, invalid run statuses, missing reads, schema errors and refusal false positives.
- [ ] Implement `scripts/benchmark.py` with explicit database guard, CLI retrieval/workflow/report stages, data validation, independent initial records, append-only partial evidence and deterministic scoring. Run failing tests then green; use real store integration in the benchmark.
- [ ] Run all retrieval methods once, then 24 real workflow cases within 96 calls. No automatic retries. Inspect every answer/proposal against facts and sources in raw outputs; write manual judgments separately without altering raw records.
- [ ] Aggregate actual denominators/intervals, document failures and environment. Test scorer and complete existing regression/lint. Commit data, harness, results as separate changes under lemon5227; push existing PR branch.
- [ ] Update the verified AI resume source and delivery PDF with measured sample scope and metrics; compile, render, inspect one-page output.
