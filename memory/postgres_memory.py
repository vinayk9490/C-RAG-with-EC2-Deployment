import os
from contextlib import contextmanager
from langgraph.checkpoint.postgres import PostgresSaver


@contextmanager
def get_postgres_checkpointer():
    """Yield a PostgresSaver checkpointer backed by the configured Postgres DB."""
    conn_string = os.environ["POSTGRES_URI"]
    with PostgresSaver.from_conn_string(conn_string) as checkpointer:
        checkpointer.setup()   # idempotent — creates tables if they don't exist
        yield checkpointer
