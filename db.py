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
            INSERT INTO semanticEdges (conversationId, fromTurnId, toTurnId, label, confidence)
            VALUES (?, ?, ?, ?, ?)
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
