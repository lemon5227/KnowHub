"""Fixed synthetic benchmark. Never resets the demo DB; never retries model failures.

Run: uv run python -m scripts.benchmark retrieval|workflow|report
Paid workflow stage is explicit, 96 provider requests maximum across 24 cases.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from dotenv import load_dotenv
from psycopg import sql

from knowhub.agent import Agent, citation_ids
from knowhub.model import Model, ModelError
from knowhub.retrieval import Embedder, Retriever
from knowhub.store import Store, now

DATA = Path(__file__).resolve().parent.parent / "benchmarks/enterprise-v1"
OUTPUT = Path("artifacts/benchmarks/enterprise-v1")
UNCERTAIN = re.compile(
    r"未找到|没有.{0,12}(依据|规定|政策|相关)|未(提供|收录|规定|检索到)|不包含|无法.{0,12}(确认|承诺|确定|提供)|无.{0,8}(依据|记录|信息)|不存在|不在.{0,8}(知识库|规范)"
)


def retrieval_metrics(ranked, relevant, ks=(1, 3, 5, 8)):
    if not relevant:
        raise ValueError("Recall needs nonempty annotated positives")
    ranked = list(dict.fromkeys(ranked))
    result = {}
    for k in ks:
        count = len(set(ranked[:k]) & relevant)
        result[f"recall@{k}"] = count / len(relevant)
        result[f"hit@{k}"] = int(count > 0)
        result[f"all_evidence@{k}"] = int(count == len(relevant))
    result["mrr@8"] = next((1 / i for i, key in enumerate(ranked[:8], 1) if key in relevant), 0)
    return result


def workflow_checks(case, run, *, unchanged):
    traces = run.get("trace", [])
    tools = {t["tool"] for t in traces if t["status"] == "ok"}
    proposal = run.get("proposal")
    answer = run.get("answer", "")
    content = proposal.get("note", "") if proposal else answer
    cited = citation_ids(content)
    known = {e["id"] for e in run.get("evidence", [])}
    kind = case["kind"]
    checks = {
        "finished": run["status"] == ("awaiting_approval" if kind == "update" else "completed"),
        "required_tools": set(case["required_tools"]) <= tools,
        "no_write_before_decision": unchanged,
        "proposal_matches_intent": bool(proposal and proposal["status"] == "in_progress")
        if kind == "update"
        else not proposal,
        "fact_terms": all(any(term in content for term in options) for options in case.get("terms", [])),
        "valid_citations": not (cited - known),
    }
    if kind in {"policy", "read", "update"}:
        checks["cites_evidence"] = bool(cited)
    if kind in {"read", "update"}:
        checks["order_identifier"] = case["order_id"] in content
    if kind in {"unknown", "missing_record"}:
        checks["explicit_uncertainty"] = bool(UNCERTAIN.search(answer))
    if kind == "missing_id":
        checks["requests_identifier"] = bool(
            re.search(r"(提供|告知|补充|需要|请问).{0,40}(编号|工单号|订单号)|请.{0,15}(工单|订单)", answer)
        )
    if kind != "missing_record":
        checks["no_tool_errors"] = all(
            t["status"] == "ok" and not (isinstance(t.get("result"), dict) and "error" in t["result"])
            for t in traces
        )
    return dict(checks, task_passed=all(checks.values()))


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic snapshot preserves earlier completed cases if interrupted during write.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def dataset():
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ")
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen data hash mismatch: " + name)
    return {
        name: json.loads((DATA / f"{name}.json").read_text()) for name in ["corpus", "queries", "workflows"]
    }


def environment():
    return {
        "started_at": now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "packages": {
            name: version(name) for name in ["torch", "sentence-transformers", "psycopg", "langgraph"]
        },
        "dataset_sha256": hashlib.sha256((DATA / "SHA256SUMS").read_bytes()).hexdigest(),
        "data": "Author-created synthetic policies, orders and tickets; not real customers or held-out public data",
    }


def benchmark_store():
    url = os.environ.get(
        "KNOWHUB_BENCHMARK_DATABASE_URL",
        "postgresql://knowhub:knowhub-local-only@localhost:55439/knowhub_benchmark",
    )
    parsed = urlsplit(url)
    if parsed.path != "/knowhub_benchmark" or parsed.query or parsed.fragment:
        raise ValueError("Benchmark only uses explicitly named knowhub_benchmark database")
    admin = urlunsplit(parsed._replace(path="/postgres"))
    with psycopg.connect(admin, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname='knowhub_benchmark'").fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier("knowhub_benchmark")))
    store = Store(url)
    store.initialize()
    return store


def retriever_for(store, data):
    embedder = Embedder()
    retriever = Retriever(store, embedder)
    existing = {d["source"] for d in store.documents()}
    for doc in data["corpus"]:
        if doc["source"] not in existing:
            retriever.ingest(doc["title"], doc["text"], doc["source"])
    return retriever


def retrieval(data, output):
    path = output / "retrieval.json"
    if path.exists():
        raise ValueError("Result exists; choose a new output directory, never overwrite a benchmark run")
    store = benchmark_store()
    retriever = retriever_for(store, data)
    rows = store.all_chunks(retriever.embedder.name)
    save(output / "corpus-index.json", {"chunks": rows})
    expected_sources = {d["source"] for d in data["corpus"]}
    if {r["source"] for r in rows} != expected_sources:
        raise ValueError("Unexpected corpus in isolated benchmark database")
    # Resolve exact annotated sections to actual production-ingestion chunk IDs.
    by_section = defaultdict(set)
    for row in rows:
        by_section[(row["source"], row["section"])].add(row["id"])
    gold = {}
    for query in data["queries"]:
        selected = [by_section[(r["source"], r["section"])] for r in query["relevant"]]
        if not all(selected):
            raise ValueError("Missing gold section: " + query["id"])
        gold[query["id"]] = set.union(*selected)
    retriever.search("初始化检索", k=8)  # One disclosed warmup, not scored.
    result = {
        "environment": environment(),
        "embedding_model": retriever.embedder.name,
        "documents": len(store.documents()),
        "chunks": len(rows),
        "warmup_queries": 1,
        "method": "Exact annotated section/chunk positives; macro recall, first positive MRR; k=8; unchanged production retriever",
        "records": [],
    }
    # Round-robin methods over the same questions. No tuned weights or rerankers.
    for query in data["queries"]:
        for mode in ["dense", "lexical", "hybrid"]:
            start = time.perf_counter()
            hits = retriever.search(query["question"], k=8, mode=mode)
            elapsed = (time.perf_counter() - start) * 1000
            result["records"].append(
                {
                    "query_id": query["id"],
                    "category": query["category"],
                    "mode": mode,
                    "question": query["question"],
                    "gold_chunk_ids": sorted(gold[query["id"]]),
                    "hits": hits,
                    "elapsed_ms": round(elapsed, 3),
                    "metrics": retrieval_metrics([h["id"] for h in hits], gold[query["id"]]),
                }
            )
        save(path, result)
        print(query["id"], "retrieval recorded", flush=True)


class ObservedModel(Model):
    """Log safe request outcome and usage. Raw provider errors never enter published output."""

    def __init__(self):
        super().__init__(session_limit=96)
        self.requests = []
        self.stop_provider = False

    def complete(self, messages, tools):
        started = time.perf_counter()
        try:
            response = super().complete(messages, tools)
        except ModelError as exc:
            self.stop_provider = any(code in str(exc) for code in ["401", "402", "403", "429"])
            self.requests.append(
                {
                    "status": "failed",
                    "request_number": self.calls,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                }
            )
            raise
        self.requests.append(
            {
                "status": "ok",
                "request_number": self.calls,
                "usage": response.get("usage", {}),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
        )
        return response


def workflow(data, output):
    path = output / "workflow.json"
    if path.exists():
        raise ValueError("Result exists; no success-only reruns/overwrites")
    store = benchmark_store()
    # Existing runs indicate a prior attempt. Never silently reset workflow state.
    if store.runs():
        raise ValueError(
            "Workflow runs already exist; inspect previous attempt rather than silently resetting"
        )
    retriever = retriever_for(store, data)
    model = ObservedModel()
    agent = Agent(store, retriever, model, max_calls=4)
    result = {
        "environment": environment(),
        "model": model.name,
        "temperature": 0,
        "request_budget": 96,
        "per_case_max_calls": 4,
        "max_completion_tokens_per_call": 1200,
        "records": [],
        "requests": [],
    }
    for case in data["workflows"]:
        with store.db() as conn:
            conn.execute(
                "INSERT INTO orders VALUES (%s,%s,%s,%s)",
                (case["order_id"], "模拟评测客户", case["order_status"], case["details"]),
            )
            conn.execute(
                "INSERT INTO tickets(id,title,customer,order_id,description) VALUES (%s,%s,%s,%s,%s)",
                (
                    case["ticket_id"],
                    "合成评测 " + case["id"],
                    "模拟评测客户",
                    case["order_id"],
                    case["details"],
                ),
            )
        before = store.ticket(case["ticket_id"])
        selected = case["ticket_id"] if case.get("select_ticket", True) else None
        run = agent.run(case["question"], ticket_id=selected)
        after_run = store.ticket(case["ticket_id"])
        record = {
            "case_id": case["id"],
            "kind": case["kind"],
            "initial_ticket": before,
            "initial_order": store.order(case["order_id"]),
            "run": run,
            "ticket_before_decision": after_run,
            "checks": workflow_checks(case, run, unchanged=before == after_run),
        }
        if case["kind"] == "update" and run["status"] == "awaiting_approval":
            approved = case["decision"] == "approve"
            try:
                decision = store.approve_run(run["id"], approved)
                once = store.ticket(case["ticket_id"])
                store.approve_run(run["id"], approved)
                twice = store.ticket(case["ticket_id"])
                audits = [a for a in store.audit() if a["run_id"] == run["id"]]
                expected = (
                    once["status"] == "in_progress"
                    and once["version"] == before["version"] + 1
                    and len(audits) == 1
                    if approved
                    else once == before and not audits
                )
                record["decision"] = {
                    "approved": approved,
                    "run_status": decision["status"],
                    "ticket": twice,
                    "audit_count": len(audits),
                    "passed": bool(expected and once == twice),
                }
            except (ValueError, KeyError):
                record["decision"] = {"approved": approved, "passed": False}
        result["records"].append(record)
        result["requests"] = model.requests
        result["attempted_requests"] = model.calls
        save(path, result)
        print(
            case["id"],
            case["kind"],
            run["status"],
            "checks=" + str(record["checks"]["task_passed"]),
            "requests=" + str(model.calls),
            flush=True,
        )
        if model.stop_provider:
            print("Stopped on provider auth/quota/rate error; partial evidence retained.", flush=True)
            break


def percentile(values, probability):
    return float(__import__("numpy").quantile(values, probability))


def rescore_payload(raw, queries, errata, chunks):
    judgments = {q["id"]: deepcopy(q["relevant"]) for q in queries}
    for change in errata["changes"]:
        if change["action"] == "replace":
            judgments[change["query_id"]] = change["relevant"]
        elif change["action"] == "add":
            judgments[change["query_id"]] += change["relevant"]
        else:
            raise ValueError("Unknown annotation correction action")
    corrected = deepcopy(raw)
    for record in corrected["records"]:
        relevant = set()
        for positive in judgments[record["query_id"]]:
            matching = {
                c["id"]
                for c in chunks
                if c["source"] == positive["source"] and c["section"] == positive["section"]
            }
            if not matching:
                raise ValueError("Corrected positive is absent from corpus index")
            relevant.update(matching)
        record["gold_chunk_ids"] = sorted(relevant)
        record["metrics"] = retrieval_metrics([h["id"] for h in record["hits"]], relevant)
    return corrected


def rescore(output):
    original = output / "retrieval.json"
    errata_path = DATA / "judgment-errata.json"
    errata = json.loads(errata_path.read_text())
    result = rescore_payload(
        json.loads(original.read_text()),
        dataset()["queries"],
        errata,
        json.loads((output / "corpus-index.json").read_text())["chunks"],
    )
    result["annotation_correction"] = {
        "version": errata["version"],
        "ranking_changed": False,
        "errata_sha256": hashlib.sha256(errata_path.read_bytes()).hexdigest(),
        "original_result_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    }
    save(output / "retrieval-corrected.json", result)


def wilson(successes, n):
    if not n:
        return None
    z = 1.96
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - half, center + half]


def summarize(output):
    retrieval_path = output / "retrieval-corrected.json"
    if not retrieval_path.exists():
        retrieval_path = output / "retrieval.json"
    retrieval_data = json.loads(retrieval_path.read_text())
    summary = {"retrieval": {}, "data": retrieval_data["environment"]["data"]}
    summary["retrieval_artifact"] = retrieval_path.name
    for mode in ["dense", "lexical", "hybrid"]:
        records = [r for r in retrieval_data["records"] if r["mode"] == mode]
        values = [r["elapsed_ms"] for r in records]
        means = {key: statistics.mean(r["metrics"][key] for r in records) for key in records[0]["metrics"]}
        summary["retrieval"][mode] = {
            "n": len(records),
            **means,
            "latency_p50_ms": statistics.median(values),
            "latency_p95_ms": percentile(values, 0.95),
            "by_category": {
                category: {
                    "n": len(group := [r for r in records if r["category"] == category]),
                    "recall@5": statistics.mean(r["metrics"]["recall@5"] for r in group),
                }
                for category in sorted({r["category"] for r in records})
            },
        }
    workflow_path = output / "workflow.json"
    if workflow_path.exists():
        data = json.loads(workflow_path.read_text())
        records = data["records"]
        passed = sum(r["checks"]["task_passed"] for r in records)
        traces = [t for r in records for t in r["run"]["trace"]]
        decisions = [r["decision"] for r in records if "decision" in r]
        summary["workflow"] = {
            "n": len(records),
            "expected_cases": 24,
            "complete": len(records) == 24,
            "mechanical_passed": passed,
            "mechanical_success_rate": passed / len(records),
            "wilson_95": wilson(passed, len(records)),
            "by_kind": {
                kind: {
                    "n": len(group := [r for r in records if r["kind"] == kind]),
                    "passed": sum(r["checks"]["task_passed"] for r in group),
                }
                for kind in sorted({r["kind"] for r in records})
            },
            "attempted_requests": data["attempted_requests"],
            "failed_requests": sum(r["status"] != "ok" for r in data["requests"]),
            "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in data["requests"]),
            "tool_calls": len(traces),
            "tool_schema_precondition_accepted": sum(t["status"] == "ok" for t in traces),
            "no_unapproved_writes": sum(r["checks"]["no_write_before_decision"] for r in records),
            "decisions": len(decisions),
            "decision_passed": sum(d["passed"] for d in decisions),
            "latency_p50_ms": statistics.median(r["run"]["elapsed_ms"] for r in records),
            "latency_p95_ms": percentile([r["run"]["elapsed_ms"] for r in records], 0.95),
        }
        manual_path = output / "manual-review.json"
        if manual_path.exists():
            review = json.loads(manual_path.read_text())
            if review["workflow_sha256"] != hashlib.sha256(workflow_path.read_bytes()).hexdigest():
                raise ValueError("Manual review refers to different model outputs")
            labels = {r["case_id"]: r["semantic_passed"] for r in review["records"]}
            if set(labels) != {r["case_id"] for r in records} or len(labels) != len(review["records"]):
                raise ValueError("Manual review must cover each case exactly once")
            passed = sum(r["checks"]["task_passed"] and labels[r["case_id"]] for r in records)
            summary["workflow"].update(
                {
                    "reviewed_passed": passed,
                    "reviewed_success_rate": passed / len(records),
                    "reviewed_wilson_95": wilson(passed, len(records)),
                }
            )
    save(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["retrieval", "workflow", "rescore", "report"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    load_dotenv()
    if args.stage == "report":
        summarize(args.output)
    elif args.stage == "rescore":
        rescore(args.output)
    else:
        {"retrieval": retrieval, "workflow": workflow}[args.stage](dataset(), args.output)


if __name__ == "__main__":
    main()
