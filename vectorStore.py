"""Turn-embedding storage backed by SQLite + the sqlite-vec extension.

The stdlib `sqlite3` build on some platforms is compiled without loadable
extension support, so this module uses `apsw` (which always allows it) and keeps
the vectors in their own database file. Everything else stays on the regular
`db.py` connection.

Embeddings are stored in a `vec0` virtual table as native `float[384]` vectors
with a cosine distance metric, so similarity is a `vec_distance_cosine(...)` call
in SQL instead of JSON text parsed and looped over in Python.
"""

from pathlib import Path

import apsw
import sqlite_vec

vectorDbPath = Path(__file__).with_name("conversationVectors.db")

EMBEDDING_DIM = 384


def _connect() -> apsw.Connection:
    connection = apsw.Connection(str(vectorDbPath))
    connection.enableloadextension(True)
    sqlite_vec.load(connection)
    connection.enableloadextension(False)
    return connection


def initVectorStore() -> None:
    connection = _connect()
    try:
        # conversationId is a filterable metadata column, not a `partition key`:
        # partitioning pre-allocates a chunk per conversation, which is huge
        # overhead for a local tool with many small conversations.
        connection.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS turnEmbeddings USING vec0(
                conversationId integer,
                turnId integer,
                embedding float[{EMBEDDING_DIM}] distance_metric=cosine
            )
            """
        )
    finally:
        connection.close()


def _serialize(embedding) -> bytes:
    return sqlite_vec.serialize_float32([float(value) for value in embedding])


def saveEmbedding(conversationId: int, turnId: int, embedding) -> None:
    connection = _connect()
    try:
        connection.execute(
            "DELETE FROM turnEmbeddings WHERE conversationId = ? AND turnId = ?",
            (conversationId, turnId),
        )
        connection.execute(
            "INSERT INTO turnEmbeddings (conversationId, turnId, embedding) VALUES (?, ?, ?)",
            (conversationId, turnId, _serialize(embedding)),
        )
    finally:
        connection.close()


def count() -> int:
    connection = _connect()
    try:
        return int(connection.execute("SELECT count(*) FROM turnEmbeddings").fetchone()[0])
    finally:
        connection.close()


def deleteConversation(conversationId: int) -> None:
    connection = _connect()
    try:
        connection.execute(
            "DELETE FROM turnEmbeddings WHERE conversationId = ?",
            (conversationId,),
        )
    finally:
        connection.close()


def turnSimilarities(conversationId: int, queryEmbedding, maxTurnId: int) -> list[tuple[int, float]]:
    """Cosine similarity of the query to every stored turn 0..maxTurnId."""
    connection = _connect()
    try:
        rows = connection.execute(
            """
            SELECT turnId, vec_distance_cosine(embedding, ?) AS distance
            FROM turnEmbeddings
            WHERE conversationId = ? AND turnId <= ?
            """,
            (_serialize(queryEmbedding), conversationId, maxTurnId),
        ).fetchall()
    finally:
        connection.close()
    return [
        (int(turnId), 1.0 - float(distance) if distance is not None else 0.0)
        for turnId, distance in rows
    ]


def turnSimilarity(conversationId: int, turnId: int, queryEmbedding) -> float | None:
    connection = _connect()
    try:
        row = connection.execute(
            """
            SELECT vec_distance_cosine(embedding, ?)
            FROM turnEmbeddings
            WHERE conversationId = ? AND turnId = ?
            """,
            (_serialize(queryEmbedding), conversationId, turnId),
        ).fetchone()
    finally:
        connection.close()
    if row is None or row[0] is None:
        return None
    return 1.0 - float(row[0])
