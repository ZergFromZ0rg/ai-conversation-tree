from conftest import chat
from conversation_tree import providers
from conversation_tree.conversationErrors import ProviderError

MESSAGES = chat(("What are cats?", "Small mammals."), ("What do cats eat?", "Meat."))


def make(client):
    return client.post(
        "/ingest/conversations", json={"source": "gemini", "externalId": "s", "messages": MESSAGES}
    ).json()["conversationId"]


def test_local_summary_is_unchanged(client):
    summary = client.get(f"/conversations/{make(client)}/summary").json()
    assert summary["mode"] == "structured-local"
    assert summary["turnCount"] == 2


def test_llm_summary_falls_back_on_the_stub_model(client):
    summary = client.post(f"/conversations/{make(client)}/summary/llm", json={}).json()
    assert summary["mode"] == "structured-local"
    assert "stub" in summary["fallbackReason"].lower()


def test_llm_summary_uses_the_model_text_and_caches(client):
    calls = []

    def fake(request):
        calls.append((request.userText, request.modelSpec))
        return "  The user asked about cats.  "

    providers.setReplier(fake)
    conversationId = make(client)
    first = client.post(f"/conversations/{conversationId}/summary/llm", json={"model": "ollama:tiny"}).json()
    assert first["mode"] == "llm" and first["model"] == "ollama:tiny"
    assert first["summary"] == "The user asked about cats."
    assert "What do cats eat?" in calls[0][0]
    client.post(f"/conversations/{conversationId}/summary/llm", json={"model": "ollama:tiny"})
    assert len(calls) == 1, "second identical request should be served from cache"


def test_llm_summary_survives_provider_failure(client):
    def boom(*args, **kwargs):
        raise ProviderError("provider down")

    providers.setReplier(boom)
    summary = client.post(f"/conversations/{make(client)}/summary/llm", json={"model": "ollama:tiny"}).json()
    assert summary["mode"] == "structured-local"
    assert "provider down" in summary["fallbackReason"]


def test_summary_validation(client):
    assert client.post("/conversations/1/summary/llm", json={"model": "nonsense"}).status_code == 422
    assert client.post("/conversations/9999/summary/llm", json={}).status_code == 404
