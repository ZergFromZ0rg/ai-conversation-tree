"""Optional PostgreSQL/pgvector ANN storage for turn embeddings.

SQLite remains the source of truth for conversation metadata and graph edges.
When PGVECTOR_DATABASE_URL is configured, this module mirrors turn embeddings
into PostgreSQL and uses an HNSW cosine index for older-turn retrieval. Every
operation is best-effort: a missing or unavailable PostgreSQL service falls
back to the local NumPy path.
"""

import logging
import os
import time

logger = logging.getLogger("vectorStore")
embeddingDimensions = 384
_schemaReady = False
_nextSchemaAttempt = 0.0
schemaRetrySeconds = 30.0
defaultDatabaseUrl = "postgresql://conversation_tree:conversation_tree@127.0.0.1:5432/conversation_tree"


def _databaseUrl() -> str | None:
    return (
        os.environ.get("PGVECTOR_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or defaultDatabaseUrl
    )


def enabled() -> bool:
    return bool(_databaseUrl())


def _connect():
    import psycopg

    return psycopg.connect(_databaseUrl())


def _embeddingLiteral(embedding) -> str:
    values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def ensureSchema() -> bool:
    global _schemaReady, _nextSchemaAttempt
    if not enabled():
        return False
    if _schemaReady:
        return True
    if time.monotonic() < _nextSchemaAttempt:
        return False
    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS conversation_vectors (
                        conversation_id BIGINT NOT NULL,
                        turn_id BIGINT NOT NULL,
                        embedding vector({embeddingDimensions}) NOT NULL,
                        PRIMARY KEY (conversation_id, turn_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS conversation_vectors_hnsw
                    ON conversation_vectors USING hnsw (embedding vector_cosine_ops)
                    """
                )
        _schemaReady = True
        _nextSchemaAttempt = 0.0
        _backfillExistingEmbeddings()
        return True
    except Exception as error:
        _nextSchemaAttempt = time.monotonic() + schemaRetrySeconds
        logger.warning("pgvector unavailable; using local vector search: %s", error)
        return False


def _backfillExistingEmbeddings() -> None:
    """Mirror embeddings already present in SQLite after first enablement."""
    try:
        import numpy as np
        from db import getConnection

        sqliteConnection = getConnection()
        try:
            rows = sqliteConnection.execute(
                "SELECT conversationId, turnId, embedding FROM turnEmbeddings"
            ).fetchall()
        finally:
            sqliteConnection.close()
        if not rows:
            return
        values = [
            (row["conversationId"], row["turnId"], _embeddingLiteral(np.frombuffer(row["embedding"], dtype=np.float32)))
            for row in rows
        ]
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO conversation_vectors (conversation_id, turn_id, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (conversation_id, turn_id)
                    DO UPDATE SET embedding = EXCLUDED.embedding
                    """,
                    values,
                )
        logger.info("mirrored %d existing embeddings to pgvector", len(values))
    except Exception as error:
        logger.warning("could not backfill existing embeddings to pgvector: %s", error)


def upsertEmbedding(conversationId: int, turnId: int, embedding) -> None:
    if not ensureSchema():
        return
    try:
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversation_vectors (conversation_id, turn_id, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (conversation_id, turn_id)
                    DO UPDATE SET embedding = EXCLUDED.embedding
                    """,
                    (conversationId, turnId, _embeddingLiteral(embedding)),
                )
    except Exception as error:
        logger.warning("could not mirror embedding to pgvector: %s", error)


def searchConversation(
    conversationId: int,
    embedding,
    excludeTurnId: int | None,
    limit: int,
) -> list[tuple[int, float]] | None:
    """Return [(turn_id, cosine_similarity)] or None when pgvector is unavailable."""
    if not ensureSchema():
        return None
    try:
        where = "conversation_id = %s"
        queryVector = _embeddingLiteral(embedding)
        parameters: list[object] = [queryVector, conversationId]
        if excludeTurnId is not None:
            where += " AND turn_id <> %s"
            parameters.append(excludeTurnId)
        parameters.append(queryVector)
        parameters.append(limit)
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT turn_id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM conversation_vectors
                    WHERE {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    parameters,
                )
                return [(int(row[0]), float(row[1])) for row in cursor.fetchall()]
    except Exception as error:
        logger.warning("pgvector search failed; using local vector search: %s", error)
        return None


def searchWorkspace(embedding, limit: int) -> list[tuple[int, int, float]] | None:
    """Search all conversations; returns (conversation_id, turn_id, score)."""
    if not ensureSchema():
        return None
    try:
        queryVector = _embeddingLiteral(embedding)
        with _connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT conversation_id, turn_id,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM conversation_vectors
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (queryVector, queryVector, limit),
                )
                return [(int(row[0]), int(row[1]), float(row[2])) for row in cursor.fetchall()]
    except Exception as error:
        logger.warning("pgvector workspace search failed; using local vector search: %s", error)
        return None
