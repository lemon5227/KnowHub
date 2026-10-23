import asyncio

from test_retrieval import FixtureEmbedder

from knowhub.mcp_server import build_mcp
from knowhub.retrieval import Retriever


def test_mcp_read_contract_uses_actual_store_and_has_no_write_tool(store):
    server = build_mcp(store, Retriever(store, FixtureEmbedder()))
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {"knowledge_search", "ticket_lookup", "order_lookup"}
    result = asyncio.run(server.call_tool("ticket_lookup", {"ticket_id": "TK-1"}))
    serialized = str(result)
    assert "ORD-1" in serialized
    assert store.ticket("TK-1")["version"] == 1
