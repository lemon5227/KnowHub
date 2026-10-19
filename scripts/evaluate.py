"""Small live capability check, not a statistically powered quality benchmark.

Runs both knowledge-only dense RAG and the native-tool agent against the same
synthetic SOP corpus/model. Tool tasks intentionally require additional business
data: compare capabilities, not retrieval algorithms. No write is approved here.
"""

import argparse
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from knowhub.agent import citation_ids

CASES = [
    {
        "id": "policy",
        "question": "订单尚未发货时，取消和退款有哪些规定？引用知识库，不要修改工单。",
        "ticket_id": None,
        "criterion": "Grounded policy, valid citations, no staged write",
    },
    {
        "id": "business_read",
        "question": "查询TK-1042关联订单的实际状态和预计延迟时间，结合规范说明如何跟进。不要更新工单。",
        "ticket_id": "TK-1042",
        "criterion": "Read real ticket+order; report48h and do not stage write",
    },
    {
        "id": "stage_update",
        "question": "核实TK-1042及其订单，按知识库延迟处理规范准备将工单改为处理中，写清依据，等我确认后再执行。",
        "ticket_id": "TK-1042",
        "criterion": "Read ticket/order/SOP; stage in_progress; no write",
    },
    {
        "id": "unknown_policy",
        "question": "公司火星运输损坏赔偿金额和承诺到账天数分别是多少？只根据内部知识库回答，不要更新工单。",
        "ticket_id": None,
        "criterion": "Explicitly report missing policy, no invented amounts or staged writes",
    },
]


def score(case, run):
    if case["id"] != "stage_update" and run["status"] != "completed":
        return False
    tools = {t["tool"] for t in run.get("trace", []) if t["status"] == "ok"}
    answer = run.get("answer", "")
    cited = citation_ids(answer)
    known = {e["id"] for e in run["evidence"]}
    if case["id"] == "policy":
        return (
            bool(cited) and cited <= known and "取消" in answer and "退款" in answer and not run["proposal"]
        )
    if case["id"] == "business_read":
        return {"ticket_lookup", "order_lookup"} <= tools and "48" in answer and not run["proposal"]
    if case["id"] == "stage_update":
        proposal = run.get("proposal") or {}
        return (
            {"ticket_lookup", "order_lookup", "knowledge_search"} <= tools
            and run["status"] == "awaiting_approval"
            and proposal.get("status") == "in_progress"
        )
    return (
        any(
            s in answer
            for s in [
                "没有",
                "未找到",
                "未检索",
                "无相关",
                "无此",
                "无法",
                "不包含",
                "未提供",
                "未收录",
                "未规定",
            ]
        )
        and not run["proposal"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8077")
    parser.add_argument("--output", default="artifacts/evaluation/live-capabilities.json")
    args = parser.parse_args()
    load_dotenv()
    records = []
    with httpx.Client(
        base_url=args.base_url,
        headers={"Authorization": "Bearer " + os.environ["KNOWHUB_API_TOKEN"]},
        timeout=150,
    ) as client:
        health = client.get("/api/health").json()
        for case in CASES:
            for mode in ["rag", "agent"]:
                response = client.post(
                    "/api/runs",
                    json={"question": case["question"], "ticket_id": case["ticket_id"], "mode": mode},
                )
                response.raise_for_status()
                run = response.json()
                passed = bool(score(case, run))
                record = {
                    "case_id": case["id"],
                    "criterion": case["criterion"],
                    "mode": mode,
                    "mechanical_check_passed": passed,
                    "run": run,
                }
                records.append(record)
                print(
                    case["id"],
                    mode,
                    run["status"],
                    "check=" + str(passed),
                    "calls=" + str(run["usage"].get("model_calls")),
                    flush=True,
                )
                # Preserve partial evidence on network failure; no success-only result selection.
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "model": health["model"],
                            "data": "synthetic SOPs and orders",
                            "method": "Four prewritten tasks per mode; mechanical checks require manual answer/source review. This compares available capabilities, not pure retrieval quality or production performance.",
                            "records": records,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()
