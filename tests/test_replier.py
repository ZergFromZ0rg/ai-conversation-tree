"""Plugging in a host's own model layer."""

import asyncio

import pytest

import conversation_tree
from conversation_tree import Settings, api, db, providers
from conversation_tree.analysisService import buildLlmSummary
from conversation_tree.chatService import (
    appendTurn,
    createConversationSession,
    getConversationSession,
    processChatMessage,
    streamChatMessage,
)
from conversation_tree.conversationErrors import InvalidInputError, ProviderError
from conversation_tree.ingestService import ingestConversation
from conftest import chat


@pytest.fixture()
def conversationId(client):
    return createConversationSession()["conversationId"]


# --- a plain function ------------------------------------------------------

def test_a_plain_function_produces_the_replies(conversationId):
    seen = []

    def reply(request):
        seen.append((request.userText, request.modelSpec, request.apiKey))
        return f"echo: {request.userText}"

    providers.setReplier(reply)
    result = processChatMessage(conversationId, "hello", "my-model", "secret")
    assert result["aiText"] == "echo: hello"
    assert seen == [("hello", "my-model", "secret")]
    # the explicit model became the conversation's default, and is used next time
    processChatMessage(conversationId, "again")
    assert seen[-1][1] == "my-model"
    assert getConversationSession(conversationId)["model"] == "my-model"


def test_with_no_model_anywhere_the_function_receives_none(conversationId):
    seen = []
    providers.setReplier(lambda request: seen.append(request.modelSpec) or "ok")
    processChatMessage(conversationId, "hello")
    assert seen == [None]


def test_a_function_gets_streaming_for_free_as_one_chunk(conversationId):
    providers.setReplier(lambda request: "whole reply")
    events = list(streamChatMessage(conversationId, "hello"))
    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["text"] == "whole reply"


def test_any_exception_from_a_function_is_a_provider_error(conversationId):
    def broken(*_args):
        raise ValueError("model offline")

    providers.setReplier(broken)
    with pytest.raises(ProviderError, match="model offline") as raised:
        processChatMessage(conversationId, "hello")
    assert isinstance(raised.value.__cause__, ValueError)


def test_a_provider_error_passes_through_unchanged(conversationId):
    original = ProviderError("quota exceeded")

    def limited(*_args):
        raise original

    providers.setReplier(limited)
    with pytest.raises(ProviderError) as raised:
        processChatMessage(conversationId, "hello")
    assert raised.value is original


def test_a_failing_replier_is_a_502_over_http(client):
    def broken(*_args):
        raise RuntimeError("boom")

    providers.setReplier(broken)
    conversationId = client.post("/conversations", json={}).json()["conversationId"]
    response = client.post(f"/conversations/{conversationId}/turns", json={"userText": "hi"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Model provider error: boom"


# --- a reply must be text ----------------------------------------------------

@pytest.mark.parametrize("reply", ["", "   \n", None, 42, ["a list"]])
def test_a_blank_or_non_text_reply_is_a_provider_error_and_stores_nothing(conversationId, reply):
    providers.setReplier(lambda request: reply)
    with pytest.raises(ProviderError, match="no text output"):
        processChatMessage(conversationId, "hello")
    assert db.getConversationTurns(conversationId) == [], "no turn may be recorded for a failed reply"


@pytest.mark.parametrize("pieces", [[], [""], ["  ", "\n"], [None], ["fine", 7]])
def test_a_blank_or_non_text_stream_is_a_provider_error_and_stores_nothing(conversationId, pieces):
    class Pieces(providers.Replier):
        def generate(self, request):
            raise AssertionError("streaming only")

        def stream(self, request):
            yield from pieces

    providers.setReplier(Pieces())
    with pytest.raises(ProviderError, match="no text output"):
        list(streamChatMessage(conversationId, "hello"))
    assert db.getConversationTurns(conversationId) == []


def test_an_empty_reply_over_http_is_a_502_not_a_stored_turn(client):
    providers.setReplier(lambda request: "")
    conversationId = client.post("/conversations", json={}).json()["conversationId"]
    response = client.post(f"/conversations/{conversationId}/turns", json={"userText": "hi"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Model provider error: Model returned no text output."
    assert client.get(f"/conversations/{conversationId}/turns").json()["turns"] == []


# --- a Replier subclass ----------------------------------------------------

class TwoModels(providers.Replier):
    modelSpecHint = "model must be 'fast' or 'smart'."

    def generate(self, request):
        return f"{request.modelSpec or 'fast'}: {request.userText}"

    def stream(self, request):
        yield from ("one ", "two ", "three")

    def isValidModelSpec(self, modelSpec):
        return modelSpec in ("fast", "smart")

    def defaultModelSpec(self):
        return "fast"


def test_a_subclass_controls_streaming_validation_and_the_default(conversationId):
    providers.setReplier(TwoModels())
    assert processChatMessage(conversationId, "hi", "smart")["aiText"] == "smart: hi"
    events = list(streamChatMessage(conversationId, "hi"))
    assert [event["text"] for event in events if event["type"] == "delta"] == ["one ", "two ", "three"]
    assert events[-1]["aiText"] == "one two three"


def test_model_validation_belongs_to_the_installed_replier(client):
    providers.setReplier(TwoModels())
    with pytest.raises(InvalidInputError, match="'fast' or 'smart'"):
        createConversationSession(None, "gpt-4")
    assert createConversationSession(None, "smart")["conversationId"]
    response = client.post("/conversations", json={"model": "gpt-4"})
    assert response.status_code == 422
    assert response.json()["detail"] == "model must be 'fast' or 'smart'."


def test_the_summary_uses_the_replier_and_its_default_model(client):
    requests = []
    providers.setReplier(lambda request: requests.append(request) or "A short summary.")
    conversationId = ingestConversation("other", "sum", chat(("What are cats?", "Small mammals.")))["conversationId"]
    summary = buildLlmSummary(conversationId)
    assert summary["mode"] == "llm" and summary["summary"] == "A short summary."
    assert requests[0].modelSpec is None
    # A summary is not a turn in the conversation, and its prompt already holds the transcript.
    assert requests[0].conversationId is None and requests[0].history == []
    providers.setReplier(lambda request: "A different summary.")
    assert buildLlmSummary(conversationId)["summary"] == "A different summary.", "a new replier must not get the old one's cached text"


# --- installing and restoring ----------------------------------------------

def test_setting_none_restores_the_builtin_providers():
    providers.setReplier(lambda *_args: "x")
    assert isinstance(providers.getReplier(), providers.CallableReplier)
    assert isinstance(providers.setReplier(None), providers.BuiltinReplier)
    assert providers.getReplier().generate(providers.ReplyRequest("hi")) == "Stub LLM response for: hi"


def test_something_that_is_not_a_replier_is_rejected():
    with pytest.raises(TypeError):
        providers.setReplier(42)


class Plain(providers.Replier):
    def generate(self, request):
        return "plain"


def test_a_replier_class_that_was_never_instantiated_is_rejected_up_front():
    with pytest.raises(TypeError, match=r"pass Plain\(\)"):
        providers.setReplier(Plain)
    with pytest.raises(TypeError, match="instance"):
        providers.setReplier(dict)  # a class is callable, but never a usable reply function
    assert isinstance(providers.getReplier(), providers.BuiltinReplier), "a rejected replier is not installed"
    assert providers.setReplier(Plain()).generate(providers.ReplyRequest("q")) == "plain"


def test_init_installs_the_replier_and_resets_it_when_omitted(reconfigurable, configured, tmp_path):
    function = lambda *_args: "x"
    conversation_tree.init(Settings(dbPath=tmp_path / "r.db"), replier=function)
    assert providers.getReplier()._function is function
    conversation_tree.init(Settings(dbPath=tmp_path / "r.db"))
    assert isinstance(providers.getReplier(), providers.BuiltinReplier)


def test_the_app_installs_its_replier_at_startup(reconfigurable, tmp_path):
    app = api.createApp(settings=Settings(dbPath=tmp_path / "a.db"), options=api.ServerOptions(), replier=TwoModels())

    async def run():
        async with app.router.lifespan_context(app):
            return providers.getReplier()

    assert isinstance(asyncio.run(run()), TwoModels)


# --- the conversation so far ------------------------------------------------

def capturing():
    requests = []

    def reply(request):
        requests.append(request)
        return f"answer {len(requests)}"

    return requests, reply


def test_each_request_carries_the_conversation_so_far(conversationId):
    requests, reply = capturing()
    providers.setReplier(reply)
    processChatMessage(conversationId, "What are cats?")
    processChatMessage(conversationId, "What do they eat?")
    processChatMessage(conversationId, "And dogs?")

    assert [request.conversationId for request in requests] == [conversationId] * 3
    assert [request.userText for request in requests] == ["What are cats?", "What do they eat?", "And dogs?"]
    assert requests[0].history == []
    assert requests[1].history == [
        {"role": "user", "content": "What are cats?"},
        {"role": "assistant", "content": "answer 1"},
    ]
    assert [message["content"] for message in requests[2].history] == [
        "What are cats?", "answer 1", "What do they eat?", "answer 2",
    ]
    assert [message["role"] for message in requests[2].history] == ["user", "assistant"] * 2


def test_streaming_requests_carry_the_history_too(conversationId):
    requests, reply = capturing()
    providers.setReplier(reply)
    list(streamChatMessage(conversationId, "What are cats?"))
    list(streamChatMessage(conversationId, "What do they eat?"))
    assert requests[0].history == []
    assert [message["content"] for message in requests[1].history] == ["What are cats?", "answer 1"]


def test_turns_a_host_recorded_itself_are_part_of_the_history(conversationId):
    requests, reply = capturing()
    providers.setReplier(reply)
    appendTurn(conversationId, "What are cats?", "Small mammals.")
    processChatMessage(conversationId, "What do they eat?")
    assert [message["content"] for message in requests[0].history] == ["What are cats?", "Small mammals."]


def test_only_turns_that_fit_the_context_window_are_sent(reconfigurable, tmp_path):
    # 400 characters is 100 tokens, so every turn below costs 200 tokens.
    conversation_tree.init(Settings(dbPath=tmp_path / "window.db", contextWindowTokens=500))
    conversationId = createConversationSession()["conversationId"]
    for number in range(4):
        appendTurn(conversationId, f"{number}" * 400, f"{number}" * 400)
    requests, reply = capturing()
    providers.setReplier(reply)
    processChatMessage(conversationId, "hi")
    # The two newest whole turns (2 and 3), nothing older: 400 of the 499 tokens left after "hi".
    assert [message["content"][0] for message in requests[0].history] == ["2", "2", "3", "3"]


def test_history_reaches_the_replier_over_http(client):
    requests, reply = capturing()
    providers.setReplier(reply)
    conversationId = client.post("/conversations", json={}).json()["conversationId"]
    client.post(f"/conversations/{conversationId}/turns", json={"userText": "What are cats?"})
    client.post(f"/conversations/{conversationId}/turns", json={"userText": "What do they eat?"})
    assert [message["content"] for message in requests[1].history] == ["What are cats?", "answer 1"]


# --- an explicit model is remembered only once the turn has worked ---------------

def failing(request):
    raise ProviderError("provider down")


def test_a_failed_turn_leaves_the_conversations_model_and_version_alone(conversationId):
    providers.setReplier(failing)
    versionBefore = db.getConversationVersion(conversationId)
    with pytest.raises(ProviderError):
        processChatMessage(conversationId, "hello", "typo-model")
    assert getConversationSession(conversationId)["model"] is None
    assert db.getConversationVersion(conversationId) == versionBefore, "a failed turn must not touch the conversation"


def test_a_failed_turn_does_not_replace_the_model_the_conversation_already_had(conversationId):
    providers.setReplier(lambda request: "ok")
    processChatMessage(conversationId, "first", "good-model")
    providers.setReplier(failing)
    with pytest.raises(ProviderError):
        processChatMessage(conversationId, "second", "other-model")
    assert getConversationSession(conversationId)["model"] == "good-model"


def test_a_blank_reply_does_not_save_the_model_either(conversationId):
    providers.setReplier(lambda request: "")
    with pytest.raises(ProviderError):
        processChatMessage(conversationId, "hello", "typo-model")
    assert getConversationSession(conversationId)["model"] is None


def test_a_stream_that_breaks_does_not_save_the_model(conversationId):
    class Breaks(providers.Replier):
        def generate(self, request):
            raise AssertionError("streaming only")

        def stream(self, request):
            yield "partial "
            raise ProviderError("connection dropped")

    providers.setReplier(Breaks())
    with pytest.raises(ProviderError):
        list(streamChatMessage(conversationId, "hello", "typo-model"))
    assert getConversationSession(conversationId)["model"] is None


def test_a_successful_turn_saves_the_explicit_model_blocking_and_streaming(conversationId):
    providers.setReplier(lambda request: "ok")
    processChatMessage(conversationId, "first", "model-a")
    assert getConversationSession(conversationId)["model"] == "model-a"
    list(streamChatMessage(conversationId, "second", "model-b"))
    assert getConversationSession(conversationId)["model"] == "model-b"
