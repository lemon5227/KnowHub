import concurrent.futures

import pytest


def staged(store, run_id="run-1"):
    run = {
        "id": run_id,
        "status": "awaiting_approval",
        "proposal": {
            "ticket_id": "TK-1",
            "status": "in_progress",
            "note": "Contact carrier",
            "expected_version": 1,
        },
    }
    store.save_run(run)
    return run


def test_stage_cannot_write_business_state(store):
    staged(store)
    assert store.ticket("TK-1")["status"] == "open"
    assert store.audit() == []


def test_parallel_duplicate_approvals_apply_exactly_once(store):
    staged(store)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: store.approve_run("run-1", True), range(2)))
    assert all(r["status"] == "completed" for r in results)
    assert store.ticket("TK-1")["version"] == 2
    assert len(store.audit()) == 1
    assert store.audit()[0]["note"] == "Contact carrier"


def test_rejected_run_cannot_later_be_approved(store):
    staged(store)
    store.approve_run("run-1", False)
    with pytest.raises(ValueError):
        store.approve_run("run-1", True)
    assert store.ticket("TK-1")["status"] == "open"


def test_stale_proposal_does_not_overwrite_newer_ticket(store):
    staged(store, "a")
    staged(store, "b")
    store.approve_run("a", True)
    with pytest.raises(ValueError, match="changed"):
        store.approve_run("b", True)
    assert len(store.audit()) == 1


def test_unsupported_status_does_not_write(store):
    run = staged(store)
    run["proposal"]["status"] = "deleted"
    store.save_run(run)
    with pytest.raises(ValueError):
        store.approve_run(run["id"], True)
    assert store.ticket("TK-1")["version"] == 1


def test_completed_result_describes_saved_state_not_pending_approval(store):
    run = staged(store)
    run["answer"] = "待人工确认，尚未修改工单。"
    store.save_run(run)
    result = store.approve_run(run["id"], True)
    assert "in_progress" in result["answer"]
    assert "待人工确认" not in result["answer"]
    assert result["draft_answer"] == "待人工确认，尚未修改工单。"
