import sqlite3

import pytest

from conftest import chat
from conversation_tree import db, ingestService, vectorStore
from conversation_tree.ingestService import ingestConversation, pairMessagesIntoTurns

CATS = chat(
    ("What are cats?", "Cats are small carnivorous mammals kept as pets."),
    ("What do cats eat?", "Cats eat meat; they are obligate carnivores."),
)


def payload(messages, externalId="abc", **extra):
    return {"source": "chatgpt", "externalId": externalId, "messages": messages, **extra}


# --- pairing (pure) -------------------------------------------------------

def test_pairing_merges_consecutive_messages_and_drops_unanswered_tail():
    turns = pairMessagesIntoTurns(
        [
            {"role": "assistant", "text": "Hi! How can I help?"},  # greeting, no prompt
            {"role": "user", "text": "first line"},
            {"role": "user", "text": "second line"},
            {"role": "assistant", "text": "part one"},
            {"role": "assistant", "text": "part two"},
            {"role": "human", "text": "next question"},
            {"role": "model", "text": "next answer"},
            {"role": "user", "text": "still being answered"},
        ]
    )
    assert turns == [
        ("first line\n\nsecond line", "part one\n\npart two"),
        ("next question", "next answer"),
    ]


def test_pairing_ignores_empty_and_unknown_roles():
    assert pairMessagesIntoTurns(
        [
            {"role": "system", "text": "be nice"},
            {"role": "user", "text": "   "},
            {"role": "user", "text": "q"},
            {"role": "tool", "text": "noise"},
            {"role": "assistant", "text": "a"},
        ]
    ) == [("q", "a")]


# --- endpoint -------------------------------------------------------------

def test_ingest_creates_a_conversation_with_turns_and_a_title(client):
    response = client.post("/ingest/conversations", json=payload(CATS, url="https://chatgpt.com/c/abc"))
    assert response.status_code == 200
    body = response.json()
    assert body["created"] is True
    assert body["addedTurns"] == 2
    assert body["turnCount"] == 2

    conversation = client.get(f"/conversations/{body['conversationId']}").json()
    assert conversation["title"] == "What are cats?"
    assert conversation["source"] == "chatgpt"
    assert conversation["sourceUrl"] == "https://chatgpt.com/c/abc"
    turns = client.get(f"/conversations/{body['conversationId']}/turns").json()["turns"]
    assert [turn["userText"] for turn in turns] == ["What are cats?", "What do cats eat?"]


def test_explicit_title_wins_over_first_message(client):
    body = client.post("/ingest/conversations", json=payload(CATS, title="Pet chat")).json()
    assert client.get(f"/conversations/{body['conversationId']}").json()["title"] == "Pet chat"


def test_resync_is_idempotent(client):
    first = client.post("/ingest/conversations", json=payload(CATS)).json()
    again = client.post("/ingest/conversations", json=payload(CATS)).json()
    assert again["conversationId"] == first["conversationId"]
    assert again["created"] is False
    assert again["addedTurns"] == 0
    assert again["rebuilt"] is False
    assert len(client.get("/conversations").json()["conversations"]) == 1


def test_resync_appends_only_the_new_turns(client):
    first = client.post("/ingest/conversations", json=payload(CATS)).json()
    grown = CATS + chat(("Do cats purr?", "Yes, cats purr when content."))
    again = client.post("/ingest/conversations", json=payload(grown)).json()
    assert again["conversationId"] == first["conversationId"]
    assert again["addedTurns"] == 1
    assert again["turnCount"] == 3
    graph = client.get(f"/conversations/{first['conversationId']}/graph").json()
    assert [node["id"] for node in graph["nodes"]] == [0, 1, 2]


def test_edited_history_rebuilds_the_conversation(client):
    first = client.post("/ingest/conversations", json=payload(CATS)).json()
    edited = chat(
        ("What are dogs?", "Dogs are domesticated canines."),
        ("What do dogs eat?", "Dogs eat meat."),
    )
    again = client.post("/ingest/conversations", json=payload(edited)).json()
    assert again["conversationId"] == first["conversationId"]
    assert again["rebuilt"] is True
    turns = client.get(f"/conversations/{first['conversationId']}/turns").json()["turns"]
    assert [turn["userText"] for turn in turns] == ["What are dogs?", "What do dogs eat?"]


# --- a rebuild is all-or-nothing ---------------------------------------------

EDITED = chat(
    ("What are dogs?", "Dogs are domesticated canines."),
    ("What do dogs eat?", "Dogs eat meat."),
)


def snapshot(conversationId):
    """Everything stored for a conversation: turns (with embeddings, edges, concepts) and edges."""
    return db.getConversationTurns(conversationId), db.listConversationEdges(conversationId)


def test_a_rebuild_that_fails_while_classifying_keeps_the_old_history(client, monkeypatch):
    conversationId = ingestConversation("chatgpt", "abc", CATS)["conversationId"]
    before = snapshot(conversationId)

    realAddTurn, calls = ingestService.addTurn, []

    def flaky(graph, userText, aiText):
        calls.append(userText)
        if len(calls) == 2:
            raise RuntimeError("model fell over")
        return realAddTurn(graph, userText, aiText)

    monkeypatch.setattr(ingestService, "addTurn", flaky)
    with pytest.raises(RuntimeError, match="fell over"):
        ingestConversation("chatgpt", "abc", EDITED)
    assert snapshot(conversationId) == before, "the old history must survive a failed rebuild"

    monkeypatch.setattr(ingestService, "addTurn", realAddTurn)
    assert ingestConversation("chatgpt", "abc", EDITED)["rebuilt"] is True, "a later sync still succeeds"


def test_a_rebuild_that_fails_while_writing_rolls_back_the_deletion(client, monkeypatch):
    conversationId = ingestConversation("chatgpt", "abc", CATS)["conversationId"]
    before = snapshot(conversationId)

    realInsert, calls = db._insertTurnBundle, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise sqlite3.OperationalError("disk full")
        return realInsert(*args, **kwargs)

    monkeypatch.setattr(db, "_insertTurnBundle", flaky)
    with pytest.raises(sqlite3.OperationalError):
        ingestConversation("chatgpt", "abc", EDITED)
    assert len(calls) == 2, "the failure happened after the old rows were deleted in the transaction"
    assert snapshot(conversationId) == before
    assert len([row for row in db.getAllTurnEmbeddings() if row["conversationId"] == conversationId]) == 2


def test_a_successful_rebuild_replaces_everything_and_leaves_nothing_behind(client):
    conversationId = ingestConversation("chatgpt", "abc", CATS)["conversationId"]
    result = ingestConversation("chatgpt", "abc", EDITED)
    assert result["rebuilt"] is True and result["addedTurns"] == 2 and result["turnCount"] == 2

    turns, _edges = snapshot(conversationId)
    assert [turn["id"] for turn in turns] == [0, 1]
    assert [turn["userText"] for turn in turns] == ["What are dogs?", "What do dogs eat?"]
    assert all(turn["embedding"] for turn in turns)
    assert len([row for row in db.getAllTurnEmbeddings() if row["conversationId"] == conversationId]) == 2


def test_a_rebuild_classifies_in_memory_and_never_asks_the_vector_index(client, monkeypatch):
    ingestConversation("chatgpt", "abc", CATS)

    def forbidden(**_kwargs):
        raise AssertionError("the index still describes the old history")

    monkeypatch.setattr(vectorStore, "searchConversation", forbidden)
    assert ingestConversation("chatgpt", "abc", EDITED)["rebuilt"] is True


def test_same_external_id_from_different_sources_stays_separate(client):
    a = client.post("/ingest/conversations", json=payload(CATS)).json()
    b = client.post("/ingest/conversations", json={**payload(CATS), "source": "claude"}).json()
    assert a["conversationId"] != b["conversationId"]


def test_ingested_turns_are_classified_like_typed_ones(client):
    body = client.post("/ingest/conversations", json=payload(CATS)).json()
    graph = client.get(f"/conversations/{body['conversationId']}/graph").json()
    assert graph["nodes"][0]["root"] is True
    assert graph["edges"], "the cat follow-up should link to the first question"


def test_ingested_turns_are_searchable(client):
    client.post("/ingest/conversations", json=payload(CATS))
    results = client.get("/search", params={"query": "obligate carnivores"}).json()["results"]
    assert results and results[0]["turnId"] == 1


def test_validation(client):
    assert client.post("/ingest/conversations", json={**payload(CATS), "source": "bing"}).status_code == 422
    assert client.post("/ingest/conversations", json=payload([])).status_code == 422
    unanswered = [{"role": "user", "text": "hello?"}]
    assert client.post("/ingest/conversations", json=payload(unanswered)).status_code == 422
    assert client.post("/ingest/conversations", json=payload(CATS, externalId="")).status_code == 422
