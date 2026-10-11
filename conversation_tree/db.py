import json
import sqlite3
import struct
import uuid

from .config import getSettings
from .conversationErrors import ConflictError


def getConnection() -> sqlite3.Connection:
    connection = sqlite3.connect(getSettings().dbPath)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initDb():
    getSettings().dbPath.parent.mkdir(parents=True, exist_ok=True)
    connection = getConnection()
    try:
        # WAL lets readers run concurrently with the single writer, so the
        # per-conversation write lock does not block graph reads.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                model TEXT,
                createdAt TEXT NOT NULL,
                updatedAt TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS turns (
                rowId INTEGER PRIMARY KEY AUTOINCREMENT,
                turnId INTEGER NOT NULL,
                conversationId INTEGER NOT NULL,
                userText TEXT NOT NULL,
                aiText TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                root INTEGER NOT NULL,
                timelineParent INTEGER,
                FOREIGN KEY (conversationId) REFERENCES conversations(id) ON DELETE CASCADE,
                UNIQUE (conversationId, turnId)
            );

            CREATE TABLE IF NOT EXISTS semanticEdges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversationId INTEGER NOT NULL,
                fromTurnId INTEGER NOT NULL,
                toTurnId INTEGER NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                origin TEXT NOT NULL DEFAULT 'auto',
                FOREIGN KEY (conversationId) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS turnConcepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversationId INTEGER NOT NULL,
                turnId INTEGER NOT NULL,
                conceptId INTEGER NOT NULL,
                conceptKey TEXT,
                FOREIGN KEY (conversationId) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS turnEmbeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversationId INTEGER NOT NULL,
                turnId INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (conversationId) REFERENCES conversations(id) ON DELETE CASCADE,
                UNIQUE (conversationId, turnId)
            );

            -- Similarity links between two concepts in different conversations,
            -- keyed by the concepts' stable conceptKeys (see turnConcepts).
            -- The pair is stored once, canonically ordered aConceptKey <=
            -- bConceptKey. No FK: a key's rows live in turnConcepts and orphans
            -- are pruned explicitly on delete / re-analysis.
            CREATE TABLE IF NOT EXISTS conceptLinks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aConceptKey TEXT NOT NULL,
                bConceptKey TEXT NOT NULL,
                score REAL NOT NULL,
                label TEXT NOT NULL,
                origin TEXT NOT NULL DEFAULT 'auto',
                updatedAt TEXT NOT NULL,
                UNIQUE (aConceptKey, bConceptKey)
            );

            CREATE INDEX IF NOT EXISTS idxTurnsConversationId
            ON turns(conversationId);

            CREATE INDEX IF NOT EXISTS idxSemanticEdgesConversationId
            ON semanticEdges(conversationId);

            CREATE INDEX IF NOT EXISTS idxTurnConceptsTurnId
            ON turnConcepts(turnId);
            """
        )

        # Migrate databases created before semanticEdges.origin existed.
        existingColumns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(semanticEdges)").fetchall()
        }
        if "origin" not in existingColumns:
            connection.execute(
                "ALTER TABLE semanticEdges ADD COLUMN origin TEXT NOT NULL DEFAULT 'auto'"
            )

        # Migrate databases created before conversations.model existed.
        conversationColumns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
        }
        if "model" not in conversationColumns:
            connection.execute("ALTER TABLE conversations ADD COLUMN model TEXT")

        # Migrate databases created before ingestion: a conversation imported
        # from elsewhere remembers where it came from so a re-sync updates it
        # instead of duplicating it.
        for column in ("source", "externalId", "sourceUrl"):
            if column not in conversationColumns:
                connection.execute(f"ALTER TABLE conversations ADD COLUMN {column} TEXT")
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idxConversationsSource
            ON conversations(source, externalId)
            WHERE source IS NOT NULL AND externalId IS NOT NULL
            """
        )

        # Migrate databases that stored embeddings as JSON text.
        embeddingColumns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(turnEmbeddings)").fetchall()
        }
        if "embeddingJson" in embeddingColumns:
            _migrateJsonEmbeddingsToBlob(connection)

        # Migrate databases created before turnConcepts.conceptKey existed.
        # Backfill one stable key per distinct (conversationId, conceptId).
        turnConceptColumns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(turnConcepts)").fetchall()
        }
        if "conceptKey" not in turnConceptColumns:
            connection.execute("ALTER TABLE turnConcepts ADD COLUMN conceptKey TEXT")
        for row in connection.execute(
            "SELECT DISTINCT conversationId, conceptId FROM turnConcepts WHERE conceptKey IS NULL"
        ).fetchall():
            connection.execute(
                """
                UPDATE turnConcepts SET conceptKey = ?
                WHERE conversationId = ? AND conceptId = ? AND conceptKey IS NULL
                """,
                (uuid.uuid4().hex, row["conversationId"], row["conceptId"]),
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idxTurnConceptsConceptKey ON turnConcepts(conceptKey)"
        )

        # Migrate conceptLinks from (conversationId, conceptId) pairs to
        # conceptKey pairs. Runs after the conceptKey backfill above.
        conceptLinkColumns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(conceptLinks)").fetchall()
        }
        if "aConceptKey" not in conceptLinkColumns:
            _migrateConceptLinksToKeys(connection)

        connection.execute("DROP INDEX IF EXISTS idxConceptLinksA")
        connection.execute("DROP INDEX IF EXISTS idxConceptLinksB")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idxConceptLinksAKey ON conceptLinks(aConceptKey)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idxConceptLinksBKey ON conceptLinks(bConceptKey)"
        )

        connection.commit()
    finally:
        connection.close()


def _migrateJsonEmbeddingsToBlob(connection: sqlite3.Connection) -> None:
    legacyRows = connection.execute(
        "SELECT conversationId, turnId, embeddingJson FROM turnEmbeddings"
    ).fetchall()

    blobRows = []
    for row in legacyRows:
        values = json.loads(row["embeddingJson"])
        blobRows.append(
            (int(row["conversationId"]), int(row["turnId"]), struct.pack(f"{len(values)}f", *values))
        )

    connection.execute("DROP TABLE turnEmbeddings")
    connection.execute(
        """
        CREATE TABLE turnEmbeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversationId INTEGER NOT NULL,
            turnId INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (conversationId) REFERENCES conversations(id) ON DELETE CASCADE,
            UNIQUE (conversationId, turnId)
        )
        """
    )
    connection.executemany(
        "INSERT INTO turnEmbeddings (conversationId, turnId, embedding) VALUES (?, ?, ?)",
        blobRows,
    )


def _migrateConceptLinksToKeys(connection: sqlite3.Connection) -> None:
    keyByConcept = {
        (int(row["conversationId"]), int(row["conceptId"])): row["conceptKey"]
        for row in connection.execute(
            "SELECT DISTINCT conversationId, conceptId, conceptKey FROM turnConcepts"
        ).fetchall()
    }
    legacyRows = connection.execute(
        """
        SELECT aConversationId, aConceptId, bConversationId, bConceptId,
               score, label, origin, updatedAt
        FROM conceptLinks
        """
    ).fetchall()

    connection.execute("DROP TABLE conceptLinks")
    connection.execute(
        """
        CREATE TABLE conceptLinks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            aConceptKey TEXT NOT NULL,
            bConceptKey TEXT NOT NULL,
            score REAL NOT NULL,
            label TEXT NOT NULL,
            origin TEXT NOT NULL DEFAULT 'auto',
            updatedAt TEXT NOT NULL,
            UNIQUE (aConceptKey, bConceptKey)
        )
        """
    )
    for row in legacyRows:
        keyA = keyByConcept.get((int(row["aConversationId"]), int(row["aConceptId"])))
        keyB = keyByConcept.get((int(row["bConversationId"]), int(row["bConceptId"])))
        if not keyA or not keyB or keyA == keyB:
            continue
        first, second = (keyA, keyB) if keyA <= keyB else (keyB, keyA)
        connection.execute(
            """
            INSERT OR IGNORE INTO conceptLinks
                (aConceptKey, bConceptKey, score, label, origin, updatedAt)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (first, second, row["score"], row["label"], row["origin"], row["updatedAt"]),
        )


def createConversation(
    title: str | None = None,
    source: str | None = None,
    externalId: str | None = None,
    sourceUrl: str | None = None,
) -> int:
    connection = getConnection()
    try:
        cursor = connection.execute(
            """
            INSERT INTO conversations (title, source, externalId, sourceUrl, createdAt, updatedAt)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (title, source, externalId, sourceUrl),
        )
        connection.commit()
        return int(cursor.lastrowid)
    except sqlite3.IntegrityError as error:
        # Only the unique (source, externalId) index can fail here.
        raise ConflictError(f"A conversation from {source!r} with id {externalId!r} already exists.") from error
    finally:
        connection.close()


def listConversations() -> list[dict]:
    connection = getConnection()
    try:
        rows = connection.execute(
            """
            SELECT id, title, model, source, externalId, sourceUrl, createdAt, updatedAt
            FROM conversations
            ORDER BY updatedAt DESC, id DESC
            """
        ).fetchall()
    finally:
        connection.close()

    return [
        {
            "id": int(row["id"]),
            "title": row["title"],
            "model": row["model"],
            "source": row["source"],
            "sourceUrl": row["sourceUrl"],
            "createdAt": str(row["createdAt"]),
            "updatedAt": str(row["updatedAt"]),
        }
        for row in rows
    ]


def getConversation(conversationId: int) -> dict | None:
    connection = getConnection()
    try:
        row = connection.execute(
            """
            SELECT id, title, model, source, externalId, sourceUrl, createdAt, updatedAt
            FROM conversations
            WHERE id = ?
            """,
            (conversationId,),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return None

    return {
        "id": int(row["id"]),
        "title": row["title"],
        "model": row["model"],
        "source": row["source"],
        "sourceUrl": row["sourceUrl"],
        "createdAt": str(row["createdAt"]),
        "updatedAt": str(row["updatedAt"]),
    }


def findConversationBySource(source: str, externalId: str) -> dict | None:
    connection = getConnection()
    try:
        row = connection.execute(
            "SELECT id FROM conversations WHERE source = ? AND externalId = ?",
            (source, externalId),
        ).fetchone()
    finally:
        connection.close()
    return getConversation(int(row["id"])) if row is not None else None


def updateConversationSource(conversationId: int, title: str | None, sourceUrl: str | None) -> None:
    """Refresh an imported conversation's title / URL without touching turns."""
    connection = getConnection()
    try:
        connection.execute(
            """
            UPDATE conversations
            SET title = COALESCE(?, title), sourceUrl = COALESCE(?, sourceUrl)
            WHERE id = ?
            """,
            (title, sourceUrl, conversationId),
        )
        connection.commit()
    finally:
        connection.close()


def _dropMirroredVectors(conversationId: int) -> None:
    """Best-effort: remove a conversation's rows from the optional pgvector index."""
    try:
        from .vectorStore import deleteConversationVectors

        deleteConversationVectors(conversationId)
    except Exception:
        pass


def getConversationVersion(conversationId: int) -> str | None:
    """Return updatedAt as a cross-process cache validation token."""
    connection = getConnection()
    try:
        row = connection.execute(
            "SELECT updatedAt FROM conversations WHERE id = ?", (conversationId,)
        ).fetchone()
    finally:
        connection.close()
    return str(row["updatedAt"]) if row is not None else None


def setConversationModel(conversationId: int, model: str | None) -> dict | None:
    connection = getConnection()
    try:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET model = ?, updatedAt = strftime('%Y-%m-%d %H:%M:%f', 'now')
            WHERE id = ?
            """,
            (model, conversationId),
        )
        connection.commit()
        if cursor.rowcount == 0:
            return None
    finally:
        connection.close()

    return getConversation(conversationId)


def setConversationTitle(conversationId: int, title: str) -> dict | None:
    connection = getConnection()
    try:
        cursor = connection.execute(
            "UPDATE conversations SET title = ?, updatedAt = strftime('%Y-%m-%d %H:%M:%f', 'now') WHERE id = ?",
            (title, conversationId),
        )
        connection.commit()
        if cursor.rowcount == 0:
            return None
    finally:
        connection.close()

    return getConversation(conversationId)


def touchConversation(connection: sqlite3.Connection, conversationId: int) -> None:
    connection.execute(
        "UPDATE conversations SET updatedAt = strftime('%Y-%m-%d %H:%M:%f', 'now') WHERE id = ?",
        (conversationId,),
    )


def deleteConversation(conversationId: int) -> bool:
    connection = getConnection()
    try:
        cursor = connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (conversationId,),
        )
        # The FK cascade above already removed this conversation's turnConcepts
        # rows, so any conceptLinks (its own 'manual' links included) pointing
        # at those now-gone conceptKeys would otherwise dangle.
        pruneOrphanConceptLinks(connection)
        connection.commit()
        deleted = cursor.rowcount > 0
    finally:
        connection.close()
    if deleted:
        _dropMirroredVectors(conversationId)
    return deleted


_TURN_TABLES = ("semanticEdges", "turnConcepts", "turnEmbeddings", "turns")


def _insertTurnBundle(
    connection: sqlite3.Connection,
    conversationId: int,
    turnId: int,
    userText: str,
    aiText: str,
    timestamp: str,
    root: bool,
    timelineParent: int | None,
    embeddingBytes: bytes,
    semanticParents: list[tuple[int, str, float]],
    conceptIds: list[int],
) -> None:
    """Write one turn's rows (turn, embedding, classifier edges, concept
    memberships) on an open connection. The caller owns the transaction."""
    connection.execute(
        """
        INSERT INTO turns (turnId, conversationId, userText, aiText, timestamp, root, timelineParent)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (turnId, conversationId, userText, aiText, timestamp, int(root), timelineParent),
    )
    connection.execute(
        "INSERT OR REPLACE INTO turnEmbeddings (conversationId, turnId, embedding) VALUES (?, ?, ?)",
        (conversationId, turnId, sqlite3.Binary(embeddingBytes)),
    )
    connection.executemany(
        """
        INSERT INTO semanticEdges (conversationId, fromTurnId, toTurnId, label, confidence, origin)
        VALUES (?, ?, ?, ?, ?, 'auto')
        """,
        [
            (conversationId, fromTurnId, turnId, label, confidence)
            for fromTurnId, label, confidence in semanticParents
        ],
    )
    # A root turn introduces a brand-new concept; inheriting turns reuse the
    # key already on that concept's earlier members.
    keyByConceptId = {
        int(row["conceptId"]): row["conceptKey"]
        for row in connection.execute(
            "SELECT DISTINCT conceptId, conceptKey FROM turnConcepts "
            "WHERE conversationId = ? AND conceptKey IS NOT NULL",
            (conversationId,),
        ).fetchall()
    }
    conceptRows = []
    for conceptId in conceptIds:
        key = keyByConceptId.get(conceptId)
        if key is None:
            key = uuid.uuid4().hex
            keyByConceptId[conceptId] = key
        conceptRows.append((conversationId, turnId, conceptId, key))
    connection.executemany(
        "INSERT INTO turnConcepts (conversationId, turnId, conceptId, conceptKey) VALUES (?, ?, ?, ?)",
        conceptRows,
    )


def _mirrorEmbedding(conversationId: int, turnId: int, embeddingBytes: bytes) -> None:
    """PostgreSQL is an optional ANN accelerator; SQLite persistence must stay
    reliable when it is not configured or temporarily offline."""
    try:
        import numpy as np
        from .vectorStore import upsertEmbedding

        upsertEmbedding(conversationId, turnId, np.frombuffer(embeddingBytes, dtype=np.float32))
    except Exception:
        pass


def saveTurnBundle(
    conversationId: int,
    turnId: int,
    userText: str,
    aiText: str,
    timestamp: str,
    root: bool,
    timelineParent: int | None,
    embeddingBytes: bytes,
    semanticParents: list[tuple[int, str, float]],
    conceptIds: list[int],
) -> None:
    """Persist one new turn with its embedding, classifier edges, and concept
    memberships in a single transaction, so a failure part-way never leaves a
    turn without its embedding or edges."""
    connection = getConnection()
    try:
        _insertTurnBundle(
            connection, conversationId, turnId, userText, aiText, timestamp, root,
            timelineParent, embeddingBytes, semanticParents, conceptIds,
        )
        touchConversation(connection, conversationId)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    _mirrorEmbedding(conversationId, turnId, embeddingBytes)


def replaceConversationTurns(conversationId: int, bundles: list[dict]) -> None:
    """Swap a conversation's entire stored history for `bundles` in one transaction.

    Each bundle holds the keyword arguments of saveTurnBundle (without the
    conversation id). Used when a re-synced chat diverged from what was stored
    (an edited prompt or regenerated answer): the new history is classified
    first, off to the side, and only then replaces the old one. All-or-nothing:
    a failure leaves the previous turns, edges, concepts, and embeddings exactly
    as they were. Concept links to the old concepts are pruned, since the
    rebuilt concepts get new keys.
    """
    connection = getConnection()
    try:
        for table in _TURN_TABLES:
            connection.execute(f"DELETE FROM {table} WHERE conversationId = ?", (conversationId,))
        for bundle in bundles:
            _insertTurnBundle(connection, conversationId, **bundle)
        pruneOrphanConceptLinks(connection)
        touchConversation(connection, conversationId)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    _dropMirroredVectors(conversationId)
    for bundle in bundles:
        _mirrorEmbedding(conversationId, bundle["turnId"], bundle["embeddingBytes"])


def applyReclassification(
    conversationId: int,
    edges: list[tuple[int, int, str, float]],
    turnMetadata: list[tuple[int, bool, int | None, list[int]]],
    conceptKeyByConceptId: dict[int, str],
) -> None:
    """Persist a full reanalysis of one conversation in a single transaction.

    Rebuilds the classifier-owned ('auto') edges and every turn's root /
    timelineParent / concept assignments. Hand-created ('manual') edges are
    left in place. Every reassigned conceptId is written with the stable
    conceptKey the caller resolved for it (inherited by turn-overlap where a
    concept survived the re-analysis, freshly minted otherwise). All-or-nothing:
    a failure rolls the whole thing back.
    """
    connection = getConnection()
    try:
        connection.execute(
            "DELETE FROM semanticEdges WHERE conversationId = ? AND origin = 'auto'",
            (conversationId,),
        )
        if edges:
            connection.executemany(
                """
                INSERT INTO semanticEdges (conversationId, fromTurnId, toTurnId, label, confidence, origin)
                VALUES (?, ?, ?, ?, ?, 'auto')
                """,
                [
                    (conversationId, fromTurnId, toTurnId, label, confidence)
                    for fromTurnId, toTurnId, label, confidence in edges
                ],
            )
        for turnId, root, timelineParent, conceptIds in turnMetadata:
            connection.execute(
                """
                UPDATE turns
                SET root = ?, timelineParent = ?
                WHERE conversationId = ? AND turnId = ?
                """,
                (int(root), timelineParent, conversationId, turnId),
            )
            connection.execute(
                "DELETE FROM turnConcepts WHERE conversationId = ? AND turnId = ?",
                (conversationId, turnId),
            )
            if conceptIds:
                connection.executemany(
                    """
                    INSERT INTO turnConcepts (conversationId, turnId, conceptId, conceptKey)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (conversationId, turnId, conceptId, conceptKeyByConceptId[conceptId])
                        for conceptId in conceptIds
                    ],
                )
        # A concept that did not survive re-analysis (matchConceptKeys gave it
        # no old key to inherit and nothing new points at its old key) leaves
        # any conceptLinks — manual ones included — dangling.
        pruneOrphanConceptLinks(connection)
        touchConversation(connection, conversationId)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def getConversationTurns(conversationId: int) -> list[dict]:
    connection = getConnection()
    try:
        rows = connection.execute(
            """
            SELECT
                turns.turnId,
                turns.conversationId,
                turns.userText,
                turns.aiText,
                turns.timestamp,
                turns.root,
                turns.timelineParent,
                turnEmbeddings.embedding
            FROM turns
            LEFT JOIN turnEmbeddings
                ON turnEmbeddings.conversationId = turns.conversationId
                AND turnEmbeddings.turnId = turns.turnId
            WHERE turns.conversationId = ?
            ORDER BY turns.turnId ASC
            """,
            (conversationId,),
        ).fetchall()

        conceptRows = connection.execute(
            """
            SELECT turnId, conceptId
            FROM turnConcepts
            WHERE conversationId = ?
            ORDER BY turnId ASC, conceptId ASC
            """,
            (conversationId,),
        ).fetchall()

        edgeRows = connection.execute(
            """
            SELECT fromTurnId, toTurnId, label, confidence
            FROM semanticEdges
            WHERE conversationId = ?
            ORDER BY toTurnId ASC, id ASC
            """,
            (conversationId,),
        ).fetchall()
    finally:
        connection.close()

    conceptIdsByTurnId: dict[int, list[int]] = {}
    for row in conceptRows:
        conceptIdsByTurnId.setdefault(int(row["turnId"]), []).append(int(row["conceptId"]))

    semanticParentsByTurnId: dict[int, list[tuple[int, str, float]]] = {}
    for row in edgeRows:
        semanticParentsByTurnId.setdefault(int(row["toTurnId"]), []).append(
            (
                int(row["fromTurnId"]),
                str(row["label"]),
                float(row["confidence"]),
            )
        )

    turns = []
    for row in rows:
        turnId = int(row["turnId"])
        turns.append(
            {
                "id": turnId,
                "conversationId": int(row["conversationId"]),
                "userText": str(row["userText"]),
                "aiText": str(row["aiText"]),
                "timestamp": str(row["timestamp"]),
                "root": bool(row["root"]),
                "timelineParent": row["timelineParent"],
                "embedding": bytes(row["embedding"]) if row["embedding"] is not None else None,
                "conceptIds": conceptIdsByTurnId.get(turnId, []),
                "semanticParents": semanticParentsByTurnId.get(turnId, []),
            }
        )

    return turns


def getConversationTurnIds(conversationId: int) -> list[int]:
    connection = getConnection()
    try:
        rows = connection.execute(
            """
            SELECT turnId
            FROM turns
            WHERE conversationId = ?
            ORDER BY turnId ASC
            """,
            (conversationId,),
        ).fetchall()
    finally:
        connection.close()

    return [int(row["turnId"]) for row in rows]


def getAllConceptMembers() -> list[dict]:
    """Every (conversation, concept) membership with its turn text and embedding.

    One row per (turn, conceptId) pair, so a turn assigned to two concepts
    appears twice. Turns without a stored embedding are skipped. Feeds the
    cross-conversation concept linker and concept labelling.
    """
    connection = getConnection()
    try:
        rows = connection.execute(
            """
            SELECT
                turnConcepts.conversationId,
                turnConcepts.conceptId,
                turnConcepts.conceptKey,
                turnConcepts.turnId,
                turns.userText,
                turns.root,
                turnEmbeddings.embedding
            FROM turnConcepts
            JOIN turns
                ON turns.conversationId = turnConcepts.conversationId
                AND turns.turnId = turnConcepts.turnId
            JOIN turnEmbeddings
                ON turnEmbeddings.conversationId = turnConcepts.conversationId
                AND turnEmbeddings.turnId = turnConcepts.turnId
            ORDER BY turnConcepts.conversationId ASC, turnConcepts.conceptId ASC
            """
        ).fetchall()
    finally:
        connection.close()

    return [
        {
            "conversationId": int(row["conversationId"]),
            "conceptId": int(row["conceptId"]),
            "conceptKey": row["conceptKey"],
            "turnId": int(row["turnId"]),
            "userText": str(row["userText"]),
            "root": bool(row["root"]),
            "embedding": bytes(row["embedding"]),
        }
        for row in rows
    ]


def getAllSearchTurns() -> list[dict]:
    """Return lightweight turn metadata used to hydrate vector search hits."""
    connection = getConnection()
    try:
        rows = connection.execute(
            """
            SELECT turns.conversationId, turns.turnId, turns.userText, turns.aiText,
                   conversations.title AS conversationTitle, conversations.source AS source
            FROM turns
            JOIN conversations ON conversations.id = turns.conversationId
            ORDER BY turns.conversationId, turns.turnId
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "conversationId": int(row["conversationId"]),
            "turnId": int(row["turnId"]),
            "userText": str(row["userText"]),
            "aiText": str(row["aiText"]),
            "conversationTitle": row["conversationTitle"],
            "source": row["source"],
        }
        for row in rows
    ]


def getAllTurnEmbeddings() -> list[dict]:
    connection = getConnection()
    try:
        rows = connection.execute(
            """
            SELECT conversationId, turnId, embedding
            FROM turnEmbeddings
            ORDER BY conversationId, turnId
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "conversationId": int(row["conversationId"]),
            "turnId": int(row["turnId"]),
            "embedding": bytes(row["embedding"]),
        }
        for row in rows
    ]


def getConceptMembership(conversationId: int) -> tuple[dict[int, set[int]], dict[int, str]]:
    """One conversation's current concept partition and its stable keys.

    Returns ({conceptId: {turnId, ...}}, {conceptId: conceptKey}). Read before a
    re-analysis so the new partition can inherit keys by turn overlap.
    """
    connection = getConnection()
    try:
        rows = connection.execute(
            "SELECT turnId, conceptId, conceptKey FROM turnConcepts WHERE conversationId = ?",
            (conversationId,),
        ).fetchall()
    finally:
        connection.close()

    membership: dict[int, set[int]] = {}
    keys: dict[int, str] = {}
    for row in rows:
        conceptId = int(row["conceptId"])
        membership.setdefault(conceptId, set()).add(int(row["turnId"]))
        if row["conceptKey"] is not None:
            keys[conceptId] = row["conceptKey"]
    return membership, keys


def listConversationEdges(conversationId: int) -> list[dict]:
    connection = getConnection()
    try:
        rows = connection.execute(
            """
            SELECT id, conversationId, fromTurnId, toTurnId, label, confidence, origin
            FROM semanticEdges
            WHERE conversationId = ?
            ORDER BY toTurnId ASC, id ASC
            """,
            (conversationId,),
        ).fetchall()
    finally:
        connection.close()

    return [
        {
            "id": int(row["id"]),
            "conversationId": int(row["conversationId"]),
            "fromTurnId": int(row["fromTurnId"]),
            "toTurnId": int(row["toTurnId"]),
            "label": str(row["label"]),
            "confidence": float(row["confidence"]),
            "origin": str(row["origin"]),
        }
        for row in rows
    ]


def createEdge(conversationId: int, fromTurnId: int, toTurnId: int, label: str, confidence: float) -> dict:
    connection = getConnection()
    try:
        cursor = connection.execute(
            """
            INSERT INTO semanticEdges (conversationId, fromTurnId, toTurnId, label, confidence, origin)
            VALUES (?, ?, ?, ?, ?, 'manual')
            """,
            (conversationId, fromTurnId, toTurnId, label, confidence),
        )
        touchConversation(connection, conversationId)
        connection.commit()
        edgeId = int(cursor.lastrowid)
    finally:
        connection.close()

    return {
        "id": edgeId,
        "conversationId": conversationId,
        "fromTurnId": fromTurnId,
        "toTurnId": toTurnId,
        "label": label,
        "confidence": confidence,
        "origin": "manual",
    }


def updateEdge(edgeId: int, label: str | None = None, confidence: float | None = None) -> dict | None:
    if label is None and confidence is None:
        return getEdge(edgeId)

    connection = getConnection()
    try:
        fields = []
        values: list[object] = []
        if label is not None:
            fields.append("label = ?")
            values.append(label)
        if confidence is not None:
            fields.append("confidence = ?")
            values.append(confidence)
        values.append(edgeId)
        row = connection.execute(
            "SELECT conversationId FROM semanticEdges WHERE id = ?", (edgeId,)
        ).fetchone()
        connection.execute(
            f"""
            UPDATE semanticEdges
            SET {", ".join(fields)}
            WHERE id = ?
            """,
            values,
        )
        if row is not None:
            touchConversation(connection, int(row["conversationId"]))
        connection.commit()
    finally:
        connection.close()

    return getEdge(edgeId)


def deleteEdge(edgeId: int) -> bool:
    connection = getConnection()
    try:
        row = connection.execute(
            "SELECT conversationId FROM semanticEdges WHERE id = ?", (edgeId,)
        ).fetchone()
        cursor = connection.execute(
            """
            DELETE FROM semanticEdges
            WHERE id = ?
            """,
            (edgeId,),
        )
        if row is not None:
            touchConversation(connection, int(row["conversationId"]))
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()


def getEdge(edgeId: int) -> dict | None:
    connection = getConnection()
    try:
        row = connection.execute(
            """
            SELECT id, conversationId, fromTurnId, toTurnId, label, confidence, origin
            FROM semanticEdges
            WHERE id = ?
            """,
            (edgeId,),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return None

    return {
        "id": int(row["id"]),
        "conversationId": int(row["conversationId"]),
        "fromTurnId": int(row["fromTurnId"]),
        "toTurnId": int(row["toTurnId"]),
        "label": str(row["label"]),
        "confidence": float(row["confidence"]),
        "origin": str(row["origin"]),
    }


def canonicalConceptKeyPair(keyA: str, keyB: str) -> tuple[str, str]:
    """Order a key pair so the smaller string comes first.

    conceptLinks stores each unordered pair once; every writer and reader uses
    this ordering so the UNIQUE constraint sees a single canonical row.
    """
    return (keyA, keyB) if keyA <= keyB else (keyB, keyA)


# Resolves each link's two conceptKeys to their current (conversationId,
# conceptId). An INNER JOIN, so a link whose concept no longer exists (dropped
# on re-analysis, or its conversation deleted) simply does not appear.
_CONCEPT_LINK_SELECT = """
    WITH concept AS (
        SELECT DISTINCT conceptKey, conversationId, conceptId
        FROM turnConcepts
        WHERE conceptKey IS NOT NULL
    )
    SELECT
        conceptLinks.id,
        conceptLinks.aConceptKey,
        conceptLinks.bConceptKey,
        conceptLinks.score,
        conceptLinks.label,
        conceptLinks.origin,
        conceptLinks.updatedAt,
        ca.conversationId AS aConversationId,
        ca.conceptId AS aConceptId,
        cb.conversationId AS bConversationId,
        cb.conceptId AS bConceptId
    FROM conceptLinks
    JOIN concept ca ON ca.conceptKey = conceptLinks.aConceptKey
    JOIN concept cb ON cb.conceptKey = conceptLinks.bConceptKey
"""


def _conceptLinkRow(row: sqlite3.Row) -> dict:
    return {
        "id": int(row["id"]),
        "aConceptKey": str(row["aConceptKey"]),
        "bConceptKey": str(row["bConceptKey"]),
        "aConversationId": int(row["aConversationId"]),
        "aConceptId": int(row["aConceptId"]),
        "bConversationId": int(row["bConversationId"]),
        "bConceptId": int(row["bConceptId"]),
        "score": float(row["score"]),
        "label": str(row["label"]),
        "origin": str(row["origin"]),
        "updatedAt": str(row["updatedAt"]),
    }


def pruneOrphanConceptLinks(connection: sqlite3.Connection | None = None) -> None:
    """Delete links whose either concept no longer exists in turnConcepts."""
    owned = connection is None
    connection = connection or getConnection()
    try:
        connection.execute(
            """
            DELETE FROM conceptLinks
            WHERE aConceptKey NOT IN (SELECT conceptKey FROM turnConcepts WHERE conceptKey IS NOT NULL)
               OR bConceptKey NOT IN (SELECT conceptKey FROM turnConcepts WHERE conceptKey IS NOT NULL)
            """
        )
        if owned:
            connection.commit()
    finally:
        if owned:
            connection.close()


def replaceAutoConceptLinks(
    conversationId: int,
    links: list[tuple[str, str, float, str]],
) -> None:
    """Rebuild the 'auto' concept links that touch one conversation.

    `links` entries are (conceptKeyA, conceptKeyB, score, label) in any order;
    they are canonicalised here. Hand-created ('manual') links are left
    untouched: the delete is scoped to origin='auto', and an insert that would
    collide with a manual row is ignored. All-or-nothing.
    """
    rows = []
    for keyA, keyB, score, label in links:
        if keyA == keyB:
            continue
        first, second = canonicalConceptKeyPair(str(keyA), str(keyB))
        rows.append((first, second, float(score), str(label)))

    connection = getConnection()
    try:
        conversationKeys = [
            row["conceptKey"]
            for row in connection.execute(
                "SELECT DISTINCT conceptKey FROM turnConcepts "
                "WHERE conversationId = ? AND conceptKey IS NOT NULL",
                (conversationId,),
            ).fetchall()
        ]
        if conversationKeys:
            placeholders = ",".join("?" * len(conversationKeys))
            connection.execute(
                f"""
                DELETE FROM conceptLinks
                WHERE origin = 'auto'
                  AND (aConceptKey IN ({placeholders}) OR bConceptKey IN ({placeholders}))
                """,
                conversationKeys + conversationKeys,
            )
        if rows:
            connection.executemany(
                """
                INSERT INTO conceptLinks
                    (aConceptKey, bConceptKey, score, label, origin, updatedAt)
                VALUES (?, ?, ?, ?, 'auto', datetime('now'))
                ON CONFLICT (aConceptKey, bConceptKey) DO NOTHING
                """,
                rows,
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def createManualConceptLink(keyA: str, keyB: str, score: float, label: str) -> dict | None:
    """Insert (or upgrade to 'manual') a hand-created link between two concepts.

    Returns the resolved link row, or None if either key is unknown.
    """
    if keyA == keyB:
        return None
    first, second = canonicalConceptKeyPair(str(keyA), str(keyB))

    connection = getConnection()
    try:
        known = {
            row["conceptKey"]
            for row in connection.execute(
                "SELECT DISTINCT conceptKey FROM turnConcepts WHERE conceptKey IN (?, ?)",
                (first, second),
            ).fetchall()
        }
        if {first, second} - known:
            return None
        connection.execute(
            """
            INSERT INTO conceptLinks (aConceptKey, bConceptKey, score, label, origin, updatedAt)
            VALUES (?, ?, ?, ?, 'manual', datetime('now'))
            ON CONFLICT (aConceptKey, bConceptKey)
            DO UPDATE SET score = excluded.score, label = excluded.label,
                          origin = 'manual', updatedAt = excluded.updatedAt
            """,
            (first, second, float(score), str(label)),
        )
        connection.commit()
        row = connection.execute(
            _CONCEPT_LINK_SELECT + " WHERE conceptLinks.aConceptKey = ? AND conceptLinks.bConceptKey = ?",
            (first, second),
        ).fetchone()
    finally:
        connection.close()

    return _conceptLinkRow(row) if row is not None else None


def deleteConceptLink(linkId: int) -> bool:
    connection = getConnection()
    try:
        cursor = connection.execute("DELETE FROM conceptLinks WHERE id = ?", (linkId,))
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()


def listConceptLinksForConversation(conversationId: int) -> list[dict]:
    connection = getConnection()
    try:
        rows = connection.execute(
            _CONCEPT_LINK_SELECT
            + """
            WHERE ca.conversationId = ? OR cb.conversationId = ?
            ORDER BY conceptLinks.score DESC, conceptLinks.id ASC
            """,
            (conversationId, conversationId),
        ).fetchall()
    finally:
        connection.close()

    return [_conceptLinkRow(row) for row in rows]


def listAllConceptLinks() -> list[dict]:
    connection = getConnection()
    try:
        rows = connection.execute(
            _CONCEPT_LINK_SELECT + " ORDER BY conceptLinks.score DESC, conceptLinks.id ASC"
        ).fetchall()
    finally:
        connection.close()

    return [_conceptLinkRow(row) for row in rows]
