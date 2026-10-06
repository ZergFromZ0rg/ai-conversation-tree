"""Regression tests for backend bugs found in the cleanup audit."""

import sqlite3
from datetime import datetime

import numpy as np
import pytest

import db
import graphService
import ingestService
import vectorStore
from conftest import chat
from graphService import ConversationGraph
from graphStore import lockedGraph
from models import TurnModel

CATS = chat(("What are cats?", "Small carnivorous mammals."))


def payload(externalId="r1", messages=CATS):
    return {"source": "chatgpt", "externalId": externalId, "messages": messages}


# --- auth responses must carry CORS headers -------------------------------

def test_401_is_readable_cross_origin(client, monkeypatch):
    monkeypatch.setenv("AI_CONVERSATION_TREE_API_TOKEN", "s3cret")
    response = client.get("/conversations", headers={"Origin": "chrome-extension://abc"})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "chrome-extension://abc"


# --- concurrent first sync of the same chat -------------------------------

def test_concurrent_first_sync_joins_the_existing_conversation(client, monkeypatch):
    first = client.post("/ingest/conversations", json=payload()).json()

    real = ingestService.findConversationBySource
    calls = []

    def raceyFind(source, externalId):
        calls.append(1)
        return None if len(calls) == 1 else real(source, externalId)  # "not there yet"

    monkeypatch.setattr(ingestService, "findConversationBySource", raceyFind)
    again = client.post("/ingest/conversations", json=payload())
    assert again.status_code == 200
    assert again.json()["conversationId"] == first["conversationId"]
    assert again.json()["created"] is False
    assert len(client.get("/conversations").json()["conversations"]) == 1


# --- graph cache must not keep a turn that failed to persist --------------

def test_failed_mutation_drops_the_cached_graph(client):
    conversationId = client.post("/conversations", json={}).json()["conversationId"]
    with pytest.raises(RuntimeError):
        with lockedGraph(conversationId) as graph:
            graph.turns.append(object())  # an in-memory turn that never reaches the DB
            raise RuntimeError("persist failed")
    with lockedGraph(conversationId) as graph:
        assert graph.turns == []


# --- a turn is written atomically ------------------------------------------

def test_turn_write_rolls_back_completely_on_failure(client):
    conversationId = client.post("/conversations", json={}).json()["conversationId"]
    vector = np.zeros(4, dtype=np.float32).tobytes()
    with pytest.raises(sqlite3.IntegrityError):
        db.saveTurnBundle(
            conversationId, 0, "q", "a", datetime.now().isoformat(), True, None, vector,
            [(0, None, 0.5)],  # NOT NULL label: fails after the turn + embedding rows
            [0],
        )
    assert db.getConversationTurns(conversationId) == []
    assert db.getAllTurnEmbeddings() == []


# --- pgvector hygiene --------------------------------------------------------

def test_stale_pgvector_candidates_are_ignored(monkeypatch):
    def vec(*values):
        return np.asarray(values, dtype=np.float32)

    graph = ConversationGraph(conversationId=1)
    for index in range(3):
        graph.turns.append(
            TurnModel(index, index == 0, vec(1, index), "q", "a", datetime.now(), None)
        )
    # id 7 / 99 exist only in the (stale) vector index, not in this graph.
    monkeypatch.setattr(
        vectorStore, "searchConversation", lambda **kw: [(99, 0.99), (7, 0.9), (0, 0.8), (1, 0.7)]
    )
    ids = [turnId for turnId, _ in graphService.retrieveCrossLinkCandidates(graph, vec(1, 0), 2)]
    assert ids and set(ids) <= {0, 1}


def test_deleting_or_rebuilding_a_conversation_drops_its_mirrored_vectors(client, monkeypatch):
    dropped = []
    monkeypatch.setattr(vectorStore, "deleteConversationVectors", dropped.append)

    first = client.post("/ingest/conversations", json=payload("v1")).json()["conversationId"]
    edited = chat(("What are dogs?", "Canines."))
    client.post("/ingest/conversations", json=payload("v1", edited))  # diverged -> rebuilt
    assert dropped == [first]
    client.delete(f"/conversations/{first}")
    assert dropped == [first, first]
