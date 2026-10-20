from fastapi.testclient import TestClient
from test_agent import ScriptedProvider, answer
from test_retrieval import FixtureEmbedder

from knowhub.agent import Agent
from knowhub.api import create_app
from knowhub.retrieval import Retriever


def client(store):
    retrieval = Retriever(store, FixtureEmbedder())
    agent = Agent(store, retrieval, ScriptedProvider([answer("请补充工单编号。")]))
    return TestClient(create_app(store, agent, "test-operator-token"))


AUTH = {"Authorization": "Bearer test-operator-token"}


def test_only_health_is_public(store):
    c = client(store)
    assert c.get("/api/health").status_code == 200
    for url in ["/api/tickets", "/api/documents", "/api/runs"]:
        assert c.get(url).status_code == 401
    assert c.post("/api/runs", json={"question": "hello"}).status_code == 401


def test_operator_can_inspect_real_tickets_and_persisted_runs(store):
    c = client(store)
    assert c.get("/api/tickets", headers=AUTH).json()[0]["id"] == "TK-1"
    r = c.post("/api/runs", headers=AUTH, json={"question": "请查询订单"})
    assert r.status_code == 200
    run = r.json()
    assert "工单编号" in run["answer"]
    assert c.get("/api/runs/" + run["id"], headers=AUTH).json()["status"] == "completed"


def test_document_upload_and_validation(store):
    c = client(store)
    r = c.post(
        "/api/documents/upload",
        headers=AUTH,
        files={"file": ("policy.md", b"# Refund\nA refund policy.", "text/markdown")},
    )
    assert r.status_code == 200
    assert r.json()["chunks"] == 1
    assert len(c.get("/api/documents", headers=AUTH).json()) == 1
    assert (
        c.post("/api/documents/upload", headers=AUTH, files={"file": ("virus.exe", b"x")}).status_code == 415
    )
    assert (
        c.post("/api/documents/upload", headers=AUTH, files={"file": ("empty.txt", b" ")}).status_code == 422
    )


def test_approval_is_protected_and_does_not_accept_arbitrary_payload(store):
    from test_store import staged

    staged(store)
    c = client(store)
    assert c.post("/api/runs/run-1/decision", json={"approved": True}).status_code == 401
    assert (
        c.post(
            "/api/runs/run-1/decision", headers=AUTH, json={"approved": True, "ticket_id": "OTHER"}
        ).status_code
        == 422
    )
    assert c.post("/api/runs/run-1/decision", headers=AUTH, json={"approved": True}).status_code == 200
    assert store.ticket("TK-1")["status"] == "in_progress"


def test_unknown_run_and_missing_ticket_are_explicit(store):
    c = client(store)
    assert c.get("/api/runs/missing", headers=AUTH).status_code == 404
    assert (
        c.post("/api/runs", headers=AUTH, json={"question": "核实", "ticket_id": "TK-missing"}).status_code
        == 404
    )
