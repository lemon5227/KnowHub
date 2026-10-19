import numpy as np

from knowhub.retrieval import Retriever


class FixtureEmbedder:
    """Independent synthetic geometry tests ranking only, not semantic quality."""

    name = "fixture-384"

    def encode(self, texts, *, query=False):
        rows = []
        for text in texts:
            row = np.zeros(384)
            row[0 if any(w in text.lower() for w in ["refund", "return"]) else 1] = 1
            rows.append(row)
        return np.array(rows)


def test_dense_retrieves_semantically_related_different_words(store):
    r = Retriever(store, FixtureEmbedder())
    r.ingest("Returns", "# Returns\nA refund is available for unused products.", "returns.md")
    r.ingest("Delivery", "# Delay\nContact the carrier for shipping delays.", "delivery.md")
    hits = r.search("return", mode="dense")
    assert hits[0]["name"] == "Returns"
    assert hits[0]["section"] == "Returns"
    assert hits[0]["source"] == "returns.md"


def test_reupload_replaces_chunks_and_keeps_source_identity(store):
    r = Retriever(store, FixtureEmbedder())
    first = r.ingest("Policy", "Old cancellation policy.", "policy.md")
    second = r.ingest("Policy", "New shipping policy.", "policy.md")
    assert first["document_id"] == second["document_id"]
    assert len(store.documents()) == 1
    assert "Old" not in str(r.search("policy"))


def test_blank_queries_and_empty_documents_do_not_create_noise(store):
    r = Retriever(store, FixtureEmbedder())
    assert r.search("  ") == []
    import pytest

    with pytest.raises(ValueError):
        r.ingest("Blank", " \n", "blank.txt")
    assert store.documents() == []


def test_hybrid_lexical_terms_and_page_metadata_survive(store):
    r = Retriever(store, FixtureEmbedder())
    r.ingest(
        "Dispatch", "# Escalation\nZX-912 escalation requires specialist support.", "dispatch.pdf", page=7
    )
    hit = r.search("ZX-912", mode="hybrid")[0]
    assert hit["page"] == 7
    assert "ZX-912" in hit["text"]
    assert hit["score"] > 0
