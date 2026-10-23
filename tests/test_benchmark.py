import pytest

from scripts.benchmark import retrieval_metrics, workflow_checks


def test_partial_gold_recall_and_duplicate_hits():
    # Mutation caught: counting duplicate hits or treating multi-evidence hit as complete recall.
    result = retrieval_metrics(["a", "a", "x", "b"], {"a", "b"}, ks=(1, 3, 5, 8))
    assert result["recall@1"] == 0.5
    assert result["recall@3"] == 1.0
    assert result["mrr@8"] == 1.0
    assert result["all_evidence@1"] == 0


def test_no_hit_is_not_success_and_empty_gold_rejected():
    assert retrieval_metrics(["x"], {"a"})["recall@5"] == 0
    assert retrieval_metrics(["x", "a"], {"a"})["mrr@8"] == 0.5
    with pytest.raises(ValueError):
        retrieval_metrics([], set())


def test_transport_failure_cannot_be_successful_unknown_policy():
    result = workflow_checks(
        {"kind": "unknown", "required_tools": []},
        {
            "status": "failed",
            "answer": "无法访问服务，未找到政策",
            "trace": [],
            "evidence": [],
            "proposal": None,
        },
        unchanged=True,
    )
    assert result["task_passed"] is False


def test_unknown_without_actual_uncertainty_fails():
    result = workflow_checks(
        {"kind": "unknown", "required_tools": []},
        {"status": "completed", "answer": "赔付金额是100元", "trace": [], "evidence": [], "proposal": None},
        unchanged=True,
    )
    assert result["task_passed"] is False


def test_proposal_requires_business_reads_and_requested_status():
    case = {
        "kind": "update",
        "required_tools": ["ticket_lookup", "order_lookup"],
        "order_id": "ORD-1",
        "terms": [["48"]],
    }
    run = {
        "status": "awaiting_approval",
        "answer": "",
        "evidence": [{"id": "a"}],
        "trace": [{"tool": "ticket_lookup", "status": "ok"}],
        "proposal": {"status": "in_progress", "note": "ORD-1 延迟48小时 [KB:a]"},
    }
    assert workflow_checks(case, run, unchanged=True)["task_passed"] is False
    run["trace"].append({"tool": "order_lookup", "status": "ok"})
    assert workflow_checks(case, run, unchanged=True)["task_passed"] is True
    assert workflow_checks(case, run, unchanged=False)["task_passed"] is False


def test_annotation_correction_preserves_original_rankings():
    from scripts.benchmark import rescore_payload

    raw = {
        "records": [{"query_id": "Q1", "hits": [{"id": "right"}], "gold_chunk_ids": ["wrong"], "metrics": {}}]
    }
    queries = [{"id": "Q1", "relevant": [{"source": "s", "section": "wrong"}]}]
    errata = {
        "changes": [
            {"query_id": "Q1", "action": "replace", "relevant": [{"source": "s", "section": "right"}]}
        ]
    }
    chunks = [{"id": "right", "source": "s", "section": "right"}]
    corrected = rescore_payload(raw, queries, errata, chunks)
    assert corrected["records"][0]["metrics"]["recall@5"] == 1
    assert corrected["records"][0]["hits"] == raw["records"][0]["hits"]
    assert raw["records"][0]["gold_chunk_ids"] == ["wrong"]


def test_benchmark_refuses_demo_database_before_connecting(monkeypatch):
    from scripts.benchmark import benchmark_store

    monkeypatch.setenv("KNOWHUB_BENCHMARK_DATABASE_URL", "postgresql://localhost/knowhub")
    with pytest.raises(ValueError, match="explicitly named knowhub_benchmark"):
        benchmark_store()
