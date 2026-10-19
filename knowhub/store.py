"""Durable data and explicit, transactional ticket decisions."""

import json
from contextlib import contextmanager
from datetime import UTC, datetime

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def now():
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, url):
        self.url = url

    @contextmanager
    def db(self):
        with psycopg.connect(self.url, row_factory=dict_row) as conn:
            register_vector(conn)
            yield conn

    def initialize(self):
        with psycopg.connect(self.url) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                  id text PRIMARY KEY, name text NOT NULL, source text UNIQUE NOT NULL,
                  embedding_model text NOT NULL, updated_at timestamptz NOT NULL DEFAULT now());
                CREATE TABLE IF NOT EXISTS chunks (
                  id text PRIMARY KEY, document_id text NOT NULL REFERENCES documents ON DELETE CASCADE,
                  section text NOT NULL, page integer, text text NOT NULL, embedding vector(384) NOT NULL);
                CREATE TABLE IF NOT EXISTS orders (
                  id text PRIMARY KEY, customer text NOT NULL, status text NOT NULL, details text NOT NULL);
                CREATE TABLE IF NOT EXISTS tickets (
                  id text PRIMARY KEY, title text NOT NULL, customer text NOT NULL,
                  order_id text REFERENCES orders, status text NOT NULL DEFAULT 'open',
                  priority text NOT NULL DEFAULT 'normal', description text NOT NULL,
                  version integer NOT NULL DEFAULT 1, resolution text NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS runs (
                  id text PRIMARY KEY, data jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
                CREATE TABLE IF NOT EXISTS audit (
                  run_id text PRIMARY KEY REFERENCES runs, ticket_id text NOT NULL REFERENCES tickets,
                  previous_status text NOT NULL, new_status text NOT NULL, note text NOT NULL,
                  created_at timestamptz NOT NULL DEFAULT now());
            """)

    def tickets(self):
        with self.db() as conn:
            return conn.execute("SELECT * FROM tickets ORDER BY id").fetchall()

    def ticket(self, ticket_id):
        with self.db() as conn:
            return conn.execute("SELECT * FROM tickets WHERE id=%s", (ticket_id,)).fetchone()

    def order(self, order_id):
        with self.db() as conn:
            return conn.execute("SELECT * FROM orders WHERE id=%s", (order_id,)).fetchone()

    def documents(self):
        with self.db() as conn:
            rows = conn.execute("""SELECT d.id,d.name,d.source,d.updated_at,count(c.id)::int AS chunks
                FROM documents d LEFT JOIN chunks c ON d.id=c.document_id
                GROUP BY d.id ORDER BY d.updated_at DESC""").fetchall()
            for row in rows:
                row["updated_at"] = row["updated_at"].isoformat()
            return rows

    def put_document(self, document_id, name, source, model, chunks, vectors):
        with self.db() as conn:
            # Serialize replacement of the same source; readers see only committed versions.
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (source,))
            conn.execute(
                """INSERT INTO documents(id,name,source,embedding_model) VALUES (%s,%s,%s,%s)
                ON CONFLICT(id) DO UPDATE SET name=EXCLUDED.name, embedding_model=EXCLUDED.embedding_model,
                updated_at=now()""",
                (document_id, name, source, model),
            )
            conn.execute("DELETE FROM chunks WHERE document_id=%s", (document_id,))
            for chunk, vector in zip(chunks, vectors, strict=True):
                conn.execute(
                    "INSERT INTO chunks VALUES (%s,%s,%s,%s,%s,%s)",
                    (chunk["id"], document_id, chunk["section"], chunk["page"], chunk["text"], vector),
                )

    def all_chunks(self, model):
        with self.db() as conn:
            return conn.execute(
                """SELECT c.id,c.document_id,c.section,c.page,c.text,d.name,d.source
                FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.embedding_model=%s""",
                (model,),
            ).fetchall()

    def dense(self, vector, model, k):
        with self.db() as conn:
            return conn.execute(
                """SELECT c.id,c.document_id,c.section,c.page,c.text,d.name,d.source,
                1-(c.embedding <=> %s) AS score FROM chunks c JOIN documents d ON d.id=c.document_id
                WHERE d.embedding_model=%s ORDER BY c.embedding <=> %s LIMIT %s""",
                (vector, model, vector, k),
            ).fetchall()

    def save_run(self, run):
        # Convert numpy / timestamps before this boundary; keep stored snapshots public and JSON-only.
        payload = json.loads(json.dumps(run))
        with self.db() as conn:
            conn.execute(
                """INSERT INTO runs(id,data) VALUES (%s,%s)
                ON CONFLICT(id) DO UPDATE SET data=EXCLUDED.data""",
                (run["id"], Jsonb(payload)),
            )

    def run(self, run_id):
        with self.db() as conn:
            row = conn.execute("SELECT data FROM runs WHERE id=%s", (run_id,)).fetchone()
            return row["data"] if row else None

    def runs(self):
        with self.db() as conn:
            return [
                r["data"] for r in conn.execute("SELECT data FROM runs ORDER BY created_at DESC LIMIT 30")
            ]

    def audit(self):
        with self.db() as conn:
            rows = conn.execute("SELECT * FROM audit ORDER BY created_at").fetchall()
            for row in rows:
                row["created_at"] = row["created_at"].isoformat()
            return rows

    def approve_run(self, run_id, approved):
        with self.db() as conn:
            row = conn.execute("SELECT data FROM runs WHERE id=%s FOR UPDATE", (run_id,)).fetchone()
            if not row:
                raise KeyError(run_id)
            run = row["data"]
            if run["status"] == "completed" and run.get("proposal") and approved:
                return run  # A repeated approval is an idempotent read.
            if run["status"] == "rejected" and not approved:
                return run
            if run["status"] != "awaiting_approval" or not run.get("proposal"):
                raise ValueError("This run has no pending decision")
            if not approved:
                run["status"] = "rejected"
            else:
                proposal = run["proposal"]
                if proposal["status"] not in {"in_progress", "resolved"} or not proposal["note"].strip():
                    raise ValueError("Invalid ticket update")
                ticket = conn.execute(
                    "SELECT * FROM tickets WHERE id=%s FOR UPDATE", (proposal["ticket_id"],)
                ).fetchone()
                if not ticket or ticket["version"] != proposal["expected_version"]:
                    raise ValueError("Ticket changed; ask for a fresh proposal")
                conn.execute(
                    """UPDATE tickets SET status=%s, resolution=%s, version=version+1 WHERE id=%s""",
                    (proposal["status"], proposal["note"], ticket["id"]),
                )
                conn.execute(
                    "INSERT INTO audit(run_id,ticket_id,previous_status,new_status,note) VALUES (%s,%s,%s,%s,%s)",
                    (run_id, ticket["id"], ticket["status"], proposal["status"], proposal["note"]),
                )
                run["status"] = "completed"
            run["decision_at"] = now()
            conn.execute("UPDATE runs SET data=%s WHERE id=%s", (Jsonb(run), run_id))
            return run
