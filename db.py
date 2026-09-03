import json
import sqlite3
from pathlib import Path


dbPath = Path(__file__).with_name("conversationTree.db")


def getConnection() -> sqlite3.Connection:
    connection = sqlite3.connect(dbPath)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initDb():
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
                FOREIGN KEY (conversationId) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS turnEmbeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversationId INTEGER NOT NULL,
                turnId INTEGER NOT NULL,
                embeddingJson TEXT NOT NULL,
                FOREIGN KEY (conversationId) REFERENCES conversations(id) ON DELETE CASCADE,
                UNIQUE (conversationId, turnId)
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

        connection.commit()
    finally:
        connection.close()


def createConversation(title: str | None = None) -> int:
    connection = getConnection()
    try:
        cursor = connection.execute(
            """
            INSERT INTO conversations (title, createdAt, updatedAt)
            VALUES (?, datetime('now'), datetime('now'))
            """,
            (title,),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def listConversations() -> list[dict]:
    connection = getConnection()
    try:
        rows = connection.execute(
            """
            SELECT id, title, createdAt, updatedAt
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
            SELECT id, title, createdAt, updatedAt
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
        "createdAt": str(row["createdAt"]),
        "updatedAt": str(row["updatedAt"]),
    }


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
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()


def saveTurn(
    conversationId: int,
    turnId: int,
    userText: str,
    aiText: str,
    timestamp: str,
    root: bool,
    timelineParent: int | None,
):
    connection = getConnection()
    try:
        connection.execute(
            """
            INSERT INTO turns (turnId, conversationId, userText, aiText, timestamp, root, timelineParent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turnId,
                conversationId,
                userText,
                aiText,
                timestamp,
                int(root),
                timelineParent,
            ),
        )
        connection.execute(
            """
            UPDATE conversations
            SET updatedAt = datetime('now')
            WHERE id = ?
            """,
            (conversationId,),
        )
        connection.commit()
    finally:
        connection.close()


def saveTurnEmbedding(conversationId: int, turnId: int, embedding) -> None:
    connection = getConnection()
    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO turnEmbeddings (conversationId, turnId, embeddingJson)
            VALUES (?, ?, ?)
            """,
            (conversationId, turnId, json.dumps(list(embedding))),
        )
        connection.commit()
    finally:
        connection.close()


def saveSemanticEdges(conversationId: int, toTurnId: int, semanticParents: list[tuple[int, str, float]]):
    connection = getConnection()
    try:
        connection.executemany(
            """
            INSERT INTO semanticEdges (conversationId, fromTurnId, toTurnId, label, confidence, origin)
            VALUES (?, ?, ?, ?, ?, 'auto')
            """,
            [
                (conversationId, fromTurnId, toTurnId, label, confidence)
                for fromTurnId, label, confidence in semanticParents
            ],
        )
        connection.commit()
    finally:
        connection.close()


def saveConceptIds(conversationId: int, turnId: int, conceptIds: list[int]):
    connection = getConnection()
    try:
        connection.executemany(
            """
            INSERT INTO turnConcepts (conversationId, turnId, conceptId)
            VALUES (?, ?, ?)
            """,
            [(conversationId, turnId, conceptId) for conceptId in conceptIds],
        )
        connection.commit()
    finally:
        connection.close()


def applyReclassification(
    conversationId: int,
    edges: list[tuple[int, int, str, float]],
    turnMetadata: list[tuple[int, bool, int | None, list[int]]],
) -> None:
    """Persist a full reanalysis of one conversation in a single transaction.

    Rebuilds the classifier-owned ('auto') edges and every turn's root /
    timelineParent / concept assignments. Hand-created ('manual') edges are
    left in place. All-or-nothing: a failure rolls the whole thing back.
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
                    INSERT INTO turnConcepts (conversationId, turnId, conceptId)
                    VALUES (?, ?, ?)
                    """,
                    [(conversationId, turnId, conceptId) for conceptId in conceptIds],
                )
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
                turnEmbeddings.embeddingJson
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
        embedding = json.loads(row["embeddingJson"]) if row["embeddingJson"] is not None else None
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
                "embedding": embedding,
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
        connection.execute(
            f"""
            UPDATE semanticEdges
            SET {", ".join(fields)}
            WHERE id = ?
            """,
            values,
        )
        connection.commit()
    finally:
        connection.close()

    return getEdge(edgeId)


def deleteEdge(edgeId: int) -> bool:
    connection = getConnection()
    try:
        cursor = connection.execute(
            """
            DELETE FROM semanticEdges
            WHERE id = ?
            """,
            (edgeId,),
        )
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

