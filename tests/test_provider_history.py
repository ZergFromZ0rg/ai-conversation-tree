"""What each built-in provider actually sends: the earlier turns, in that provider's own format.

No network: the HTTP calls are replaced by fakes that record the request."""

import pytest

from conversation_tree import providers
from conversation_tree.chatService import appendTurn, createConversationSession, processChatMessage
from conversation_tree.providers import ReplyRequest, generateAiText, streamAiText

HISTORY = [
    {"role": "user", "content": "What are cats?"},
    {"role": "assistant", "content": "Small mammals."},
]
QUESTION = "What do they eat?"


class FakeResponse:
    def __init__(self, body=None):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body

    def iter_lines(self):
        return iter(())


BODIES = {
    "ollama": {"message": {"content": "ok"}},
    "openai": {"output": [{"content": [{"type": "output_text", "text": "ok"}]}]},
    "anthropic": {"content": [{"type": "text", "text": "ok"}]},
    "gemini": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
}


@pytest.fixture()
def sent(monkeypatch):
    """Records the most recent request (blocking or streaming) and answers like a provider."""
    record = {}

    def post(url, **kwargs):
        record.update(url=url, **kwargs)
        provider = next(name for name in BODIES if name in url or (name == "gemini" and "googleapis" in url))
        return FakeResponse(BODIES[provider])

    class FakeStream:
        def __init__(self, url, kwargs):
            record.update(url=url, **kwargs)

        def __enter__(self):
            return FakeResponse()

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(providers.httpx, "post", post)
    monkeypatch.setattr(providers.httpx, "stream", lambda method, url, **kwargs: FakeStream(url, kwargs))
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test")
    return record


# --- blocking requests ------------------------------------------------------

def test_ollama_receives_history_then_the_question(sent):
    assert generateAiText(QUESTION, "ollama:tiny", history=HISTORY) == "ok"
    assert sent["json"]["messages"] == [*HISTORY, {"role": "user", "content": QUESTION}]


def test_anthropic_receives_history_then_the_question(sent):
    generateAiText(QUESTION, "anthropic:claude-test", "key", HISTORY)
    assert sent["json"]["messages"] == [*HISTORY, {"role": "user", "content": QUESTION}]


def test_openai_receives_history_as_plain_text_then_the_question(sent):
    generateAiText(QUESTION, "openai:gpt-test", "key", HISTORY)
    assert sent["json"]["input"] == [
        {"role": "user", "content": "What are cats?"},
        {"role": "assistant", "content": "Small mammals."},
        {"role": "user", "content": [{"type": "input_text", "text": QUESTION}]},
    ]


def test_gemini_calls_the_assistant_role_model(sent):
    generateAiText(QUESTION, "gemini:flash", "key", HISTORY)
    assert sent["json"]["contents"] == [
        {"role": "user", "parts": [{"text": "What are cats?"}]},
        {"role": "model", "parts": [{"text": "Small mammals."}]},
        {"role": "user", "parts": [{"text": QUESTION}]},
    ]


@pytest.mark.parametrize("spec", ["ollama:tiny", "anthropic:claude-test", "openai:gpt-test", "gemini:flash"])
def test_a_first_turn_sends_just_the_question(sent, spec):
    generateAiText(QUESTION, spec, "key")
    payload = sent["json"]
    messages = payload.get("messages") or payload.get("input") or payload.get("contents")
    assert len(messages) == 1 and messages[0]["role"] == "user"


def test_the_stub_ignores_history(sent):
    assert generateAiText(QUESTION, "stub", history=HISTORY) == f"Stub LLM response for: {QUESTION}"
    assert sent == {}, "the stub makes no HTTP call"


# --- streaming requests -----------------------------------------------------

def test_ollama_stream_receives_history(sent):
    assert list(streamAiText(QUESTION, "ollama:tiny", history=HISTORY)) == []
    assert sent["json"]["stream"] is True
    assert sent["json"]["messages"] == [*HISTORY, {"role": "user", "content": QUESTION}]


def test_anthropic_stream_receives_history(sent):
    list(streamAiText(QUESTION, "anthropic:claude-test", "key", HISTORY))
    assert sent["json"]["messages"] == [*HISTORY, {"role": "user", "content": QUESTION}]


def test_openai_stream_receives_history(sent):
    list(streamAiText(QUESTION, "openai:gpt-test", "key", HISTORY))
    assert [item["role"] for item in sent["json"]["input"]] == ["user", "assistant", "user"]


def test_gemini_streams_through_the_blocking_call_with_history(sent):
    assert list(streamAiText(QUESTION, "gemini:flash", "key", HISTORY)) == ["ok"]
    assert [item["role"] for item in sent["json"]["contents"]] == ["user", "model", "user"]


# --- from a request object, and through the real chat flow --------------------

def test_the_builtin_replier_forwards_everything_in_the_request(sent):
    request = ReplyRequest(QUESTION, "ollama:tiny", None, conversationId=7, history=HISTORY)
    assert providers.BuiltinReplier().generate(request) == "ok"
    assert sent["json"]["messages"] == [*HISTORY, {"role": "user", "content": QUESTION}]


def test_a_real_conversation_sends_its_earlier_turns_to_the_model(client, sent):
    conversationId = createConversationSession()["conversationId"]
    processChatMessage(conversationId, "What are cats?", "ollama:tiny")
    first = sent["json"]["messages"]
    assert first == [{"role": "user", "content": "What are cats?"}]

    processChatMessage(conversationId, QUESTION)
    assert sent["json"]["messages"] == [
        {"role": "user", "content": "What are cats?"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": QUESTION},
    ]


def test_turns_recorded_with_appendTurn_are_sent_too(client, sent):
    conversationId = createConversationSession()["conversationId"]
    appendTurn(conversationId, "What are cats?", "Small mammals.")
    processChatMessage(conversationId, QUESTION, "ollama:tiny")
    assert sent["json"]["messages"] == [*HISTORY, {"role": "user", "content": QUESTION}]
