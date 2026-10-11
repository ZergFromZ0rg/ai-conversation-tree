"""The pre-existing API surface: conversations, turns, edges, reports, concept links."""

import json

import httpx
import pytest

import chatService


def newConversation(client, title=None, model=None):
    body = {"title": title, "model": model}
    response = client.post("/conversations", json=body)
    assert response.status_code == 201
    return response.json()["conversationId"]


def send(client, conversationId, text):
    response = client.post(f"/conversations/{conversationId}/turns", json={"userText": text})
    assert response.status_code == 201, response.text
    return response.json()


# --- conversations --------------------------------------------------------

def test_conversation_crud(client):
    conversationId = newConversation(client, "First")
    assert client.get(f"/conversations/{conversationId}").json()["title"] == "First"
    assert [c["id"] for c in client.get("/conversations").json()["conversations"]] == [conversationId]
    assert client.delete(f"/conversations/{conversationId}").status_code == 204
    assert client.get(f"/conversations/{conversationId}").status_code == 404
    assert client.delete(f"/conversations/{conversationId}").status_code == 404


def test_model_spec_validation_and_patch(client):
    assert client.post("/conversations", json={"model": "bogus"}).status_code == 422
    conversationId = newConversation(client)
    patched = client.patch(f"/conversations/{conversationId}", json={"model": "ollama:tiny"})
    assert patched.status_code == 200 and patched.json()["model"] == "ollama:tiny"
    assert client.patch(f"/conversations/{conversationId}", json={"model": "x:y"}).status_code == 422
    assert client.patch("/conversations/9999", json={"model": "stub"}).status_code == 404


# --- turns ----------------------------------------------------------------

def test_first_turn_titles_an_untitled_conversation_only(client):
    untitled = newConversation(client)
    titled = newConversation(client, "Keep me")
    send(client, untitled, "What are cats?")
    send(client, titled, "What are cats?")
    assert client.get(f"/conversations/{untitled}").json()["title"] == "What are cats?"
    assert client.get(f"/conversations/{titled}").json()["title"] == "Keep me"


def test_turns_are_persisted_and_graphed(client):
    conversationId = newConversation(client)
    send(client, conversationId, "What are cats?")
    reply = send(client, conversationId, "What do cats eat?")
    assert reply["turnId"] == 1
    graph = client.get(f"/conversations/{conversationId}/graph").json()
    assert len(graph["nodes"]) == 2
    assert graph["nodes"][0]["root"] is True


def test_turn_rejects_empty_text_and_unknown_conversation(client):
    conversationId = newConversation(client)
    assert client.post(f"/conversations/{conversationId}/turns", json={"userText": ""}).status_code == 422
    assert client.post("/conversations/9999/turns", json={"userText": "hi"}).status_code == 404


def test_streaming_turn_emits_deltas_then_done(client):
    conversationId = newConversation(client)
    response = client.post(f"/conversations/{conversationId}/turns/stream", json={"userText": "What are cats?"})
    events = [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]
    assert events[0]["type"] == "delta"
    assert events[-1]["type"] == "done" and events[-1]["turnId"] == 0
    assert "".join(e["text"] for e in events if e["type"] == "delta") == events[-1]["aiText"]


def test_analyze_rebuilds_and_keeps_manual_edges(client):
    conversationId = newConversation(client)
    for text in ("What are cats?", "What is the capital of France?", "What do cats eat?"):
        send(client, conversationId, text)
    manual = client.post(
        "/edges",
        json={"conversationId": conversationId, "fromTurnId": 0, "toTurnId": 1, "label": "related", "confidence": 0.9},
    ).json()
    client.post(f"/conversations/{conversationId}/analyze")
    edges = client.get(f"/conversations/{conversationId}/graph").json()["edges"]
    assert any(edge["origin"] == "manual" and edge["id"] == manual["id"] for edge in edges)


# --- edges ----------------------------------------------------------------

def test_edge_lifecycle_and_validation(client):
    conversationId = newConversation(client)
    send(client, conversationId, "What are cats?")
    send(client, conversationId, "What is the capital of France?")
    base = {"conversationId": conversationId, "fromTurnId": 0, "toTurnId": 1, "label": "branch", "confidence": 0.8}

    assert client.post("/edges", json={**base, "label": "nope"}).status_code == 422
    assert client.post("/edges", json={**base, "confidence": 1.5}).status_code == 422
    assert client.post("/edges", json={**base, "fromTurnId": 1, "toTurnId": 0}).status_code == 422
    assert client.post("/edges", json={**base, "fromTurnId": 0, "toTurnId": 0}).status_code == 422
    assert client.post("/edges", json={**base, "toTurnId": 7}).status_code == 404
    assert client.post("/edges", json={**base, "conversationId": 9999}).status_code == 404

    edge = client.post("/edges", json=base).json()
    assert edge["origin"] == "manual"
    patched = client.patch(f"/edges/{edge['id']}", json={"label": "continuation"})
    assert patched.status_code == 200 and patched.json()["label"] == "continuation"
    assert client.patch(f"/edges/{edge['id']}", json={"label": "nope"}).status_code == 422
    assert client.delete(f"/edges/{edge['id']}").status_code == 204
    assert client.delete(f"/edges/{edge['id']}").status_code == 404


# --- reports --------------------------------------------------------------

def test_summary_and_reports(client):
    conversationId = newConversation(client)
    send(client, conversationId, "What are cats?")
    summary = client.get(f"/conversations/{conversationId}/summary").json()
    assert summary["turnCount"] == 1 and summary["threadCount"] == 1

    markdown = client.get(f"/conversations/{conversationId}/report")
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "# What are cats?" in markdown.text
    report = client.get(f"/conversations/{conversationId}/report", params={"format": "json"}).json()
    assert report["turns"][0]["userText"] == "What are cats?"
    assert client.get(f"/conversations/{conversationId}/report", params={"format": "pdf"}).status_code == 422
    assert client.get("/conversations/9999/report").status_code == 404


# --- concept links --------------------------------------------------------

def test_concept_link_validation(client):
    assert client.post("/concept-links", json={"aConceptKey": "a", "bConceptKey": "a"}).status_code == 422
    assert client.post("/concept-links", json={"aConceptKey": "a", "bConceptKey": "b", "kind": "x"}).status_code == 422
    assert client.post("/concept-links", json={"aConceptKey": "a", "bConceptKey": "b"}).status_code == 404
    assert client.delete("/concept-links/9999").status_code == 404


def test_similar_conversations_get_linked_and_manual_links_work(client):
    a = newConversation(client, "A")
    b = newConversation(client, "B")
    for conversationId in (a, b):
        send(client, conversationId, "How do Python decorators work?")
    client.post("/concepts/relink")
    graph = client.get("/concepts/graph").json()
    assert graph["edges"], "identical questions in two chats should link"
    nodes = {node["conversationId"]: node for node in graph["nodes"]}

    c = newConversation(client, "C")
    send(client, c, "What is the boiling point of water?")
    other = next(n for n in client.get("/concepts/graph").json()["nodes"] if n["conversationId"] == c)
    link = client.post(
        "/concept-links",
        json={"aConceptKey": nodes[a]["conceptKey"], "bConceptKey": other["conceptKey"], "kind": "related"},
    )
    assert link.status_code == 201 and link.json()["origin"] == "manual"
    client.post("/concepts/relink")
    origins = {edge["origin"] for edge in client.get("/concepts/graph").json()["edges"]}
    assert "manual" in origins, "manual links survive an auto relink"


def test_root_and_models_endpoints(client):
    assert "POST /ingest/conversations" in client.get("/").json()["endpoints"]
    models = client.get("/models").json()
    assert "stub" in [model["id"] for model in models["models"]]


# --- error contract: domain exceptions map to status codes -----------------

def test_provider_failure_is_a_502_and_a_stray_runtime_error_is_not(client, monkeypatch):
    conversationId = newConversation(client)

    def unreachable(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(chatService, "generateOllamaText", unreachable)
    response = client.post(f"/conversations/{conversationId}/turns", json={"userText": "hi", "model": "ollama:tiny"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Model provider error: connection refused"

    # An unrelated RuntimeError (a bug) must surface as itself, not as a "provider error".
    def broken(*_args, **_kwargs):
        raise RuntimeError("classifier bug")

    monkeypatch.setattr(chatService, "addTurn", broken)
    with pytest.raises(RuntimeError, match="classifier bug"):
        # Pinned to stub: the failed request above stored ollama:tiny as the conversation default.
        client.post(f"/conversations/{conversationId}/turns", json={"userText": "hi", "model": "stub"})


def test_streaming_a_bad_request_is_a_real_http_error_not_an_event(client):
    conversationId = newConversation(client)
    assert client.post("/conversations/9999/turns/stream", json={"userText": "hi"}).status_code == 404
    bad = client.post(f"/conversations/{conversationId}/turns/stream", json={"userText": "hi", "model": "bogus"})
    assert bad.status_code == 422
