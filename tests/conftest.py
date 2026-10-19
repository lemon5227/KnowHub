import os

import psycopg
import pytest

from knowhub.store import Store


@pytest.fixture
def store():
    url = os.environ.get(
        "KNOWHUB_TEST_DATABASE_URL", "postgresql://knowhub:knowhub-local-only@localhost:55439/knowhub_test"
    )
    if not url.endswith("/knowhub_test"):
        raise RuntimeError("Tests only reset a dedicated knowhub_test database")
    result = Store(url)
    result.initialize()
    with psycopg.connect(url) as conn:
        conn.execute("TRUNCATE audit, runs, chunks, documents, tickets, orders CASCADE")
        conn.execute(
            "INSERT INTO orders (id,customer,status,details) VALUES ('ORD-1','Demo','delayed','Carrier delayed by 48 hours')"
        )
        conn.execute(
            "INSERT INTO tickets (id,title,customer,order_id,description) VALUES ('TK-1','Delay','Demo','ORD-1','Delivery delayed')"
        )
    return result
