import json

import pytest

from knowhub.agent import Agent


class ScriptedProvider:
    """Only remote model output is replaced; graph, tools and DB remain real."""

    name = "fixture-model"

    def __init__(self, messages):
        self.messages = iter(messages)

    def complete(self, messages, tools):
        return {"message": next(self.messages), "usage": {"prompt_tokens": 10, "completion_tokens": 4}}


class NoKnowledge:
    def search(self, *args, **kwargs):
        return []


def call(name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def answer(text):
    return {"role": "assistant", "content": text}


def test_native_read_and_stage_require_confirmation(store):
    provider = ScriptedProvider(
        [
            call("ticket_lookup", {"ticket_id": "TK-1"}),
            call(
                "propose_ticket_update",
                {"ticket_id": "TK-1", "status": "in_progress", "note": "Contact carrier"},
            ),
        ]
    )
    run = Agent(store, NoKnowledge(), provider).run("Inspect TK-1 and prepare an update", "TK-1")
    assert run["status"] == "awaiting_approval"
    assert run["proposal"]["expected_version"] == 1
    assert store.ticket("TK-1")["status"] == "open"
    store.approve_run(run["id"], True)
    assert store.ticket("TK-1")["status"] == "in_progress"


@pytest.mark.parametrize(
    "name,args",
    [
        ("execute_sql", {"query": "DELETE FROM tickets"}),
        ("ticket_lookup", {"ticket_id": "TK-1", "unexpected": "x"}),
        ("propose_ticket_update", {"ticket_id": "TK-1", "status": "resolved", "note": "unchecked"}),
    ],
)
def test_invalid_or_unobserved_tools_do_not_modify_business_data(store, name, args):
    provider = ScriptedProvider([call(name, args), answer("Cannot complete")])
    run = Agent(store, NoKnowledge(), provider).run("Do it", "TK-1")
    assert run["trace"][0]["status"] == "error"
    assert store.ticket("TK-1")["status"] == "open"
    assert run["proposal"] is None


def test_exhausted_model_budget_is_reported_and_persisted(store):
    provider = ScriptedProvider([call("ticket_lookup", {"ticket_id": "TK-1"})] * 4)
    run = Agent(store, NoKnowledge(), provider, max_calls=2).run("Keep checking", "TK-1")
    assert run["status"] == "exhausted"
    assert len(run["trace"]) == 2
    assert store.run(run["id"])["status"] == "exhausted"


def test_model_failure_is_explicit_and_does_not_save_secrets(store):
    class Broken:
        name = "broken"

        def complete(self, *args):
            raise RuntimeError("PRIVATE_KEY_SHOULD_NOT_BE_STORED")

    run = Agent(store, NoKnowledge(), Broken()).run("hello")
    assert run["status"] == "failed"
    assert "PRIVATE_KEY" not in json.dumps(store.run(run["id"]))


def test_fabricated_source_id_cannot_be_presented_as_verified(store):
    p = ScriptedProvider([answer("Refund approved [KB:fake-document]")])
    run = Agent(store, NoKnowledge(), p).run("Refund policy?")
    assert run["status"] == "failed"
    assert "Refund approved" not in run["answer"]


def test_rag_mode_cannot_expose_business_tools(store):
    p = ScriptedProvider([call("order_lookup", {"order_id": "ORD-1"}), answer("Cannot access")])
    run = Agent(store, NoKnowledge(), p).run("Check order", mode="rag")
    assert run["trace"][0]["status"] == "error"


def test_empty_provider_answer_is_failed_not_completed(store):
    run = Agent(store, NoKnowledge(), ScriptedProvider([answer("")])).run("hello")
    assert run["status"] == "failed"


def test_forged_citation_in_staged_note_is_rejected(store):
    p = ScriptedProvider(
        [
            call("ticket_lookup", {"ticket_id": "TK-1"}),
            call(
                "propose_ticket_update",
                {"ticket_id": "TK-1", "status": "in_progress", "note": "Policy [KB:forged]"},
            ),
            answer("来源无法核实。"),
        ]
    )
    run = Agent(store, NoKnowledge(), p).run("Prepare update", "TK-1")
    assert run["proposal"] is None
    assert store.ticket("TK-1")["version"] == 1


def test_rag_embedding_failure_is_persisted_as_failure(store):
    class BrokenRetrieval:
        def search(self, *args, **kwargs):
            raise RuntimeError("Private transport context")

    run = Agent(store, BrokenRetrieval(), ScriptedProvider([])).run("policy", mode="rag")
    assert run["status"] == "failed"
    assert store.run(run["id"])["status"] == "failed"


def test_pending_decision_is_published_only_after_final_snapshot(store):
    """An operator must never approve a snapshot that an agent later overwrites."""
    provider = ScriptedProvider(
        [
            call("ticket_lookup", {"ticket_id": "TK-1"}),
            call(
                "propose_ticket_update",
                {"ticket_id": "TK-1", "status": "in_progress", "note": "Contact carrier"},
            ),
        ]
    )
    original_save = store.save_run
    pending_saves = []

    def approve_when_published(run):
        original_save(run)
        if run["status"] == "awaiting_approval":
            pending_saves.append(run["id"])
            store.approve_run(run["id"], True)

    store.save_run = approve_when_published
    run = Agent(store, NoKnowledge(), provider).run("Prepare update", "TK-1")
    assert len(pending_saves) == 1
    assert store.run(run["id"])["status"] == "completed"
    assert len(store.audit()) == 1


def test_multiple_valid_citations_in_one_bracket_are_accepted(store):
    class Sources:
        def search(self, *args, **kwargs):
            return [{"id": "source-a"}, {"id": "source-b"}]

    p = ScriptedProvider([answer("按两份规范处理 [KB:source-a, KB:source-b]")])
    run = Agent(store, Sources(), p).run("policy", mode="rag")
    assert run["status"] == "completed"
