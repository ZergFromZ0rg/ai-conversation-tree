"""Plugging in a host's own model layer."""

import asyncio

import pytest

import conversation_tree
from conversation_tree import Settings, api, providers
from conversation_tree.analysisService import buildLlmSummary
from conversation_tree.chatService import (
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

    def reply(userText, modelSpec, apiKey):
        seen.append((userText, modelSpec, apiKey))
        return f"echo: {userText}"

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
    providers.setReplier(lambda text, spec, key: seen.append(spec) or "ok")
    processChatMessage(conversationId, "hello")
    assert seen == [None]


def test_a_function_gets_streaming_for_free_as_one_chunk(conversationId):
    providers.setReplier(lambda text, spec, key: "whole reply")
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


# --- a Replier subclass ----------------------------------------------------

class TwoModels(providers.Replier):
    modelSpecHint = "model must be 'fast' or 'smart'."

    def generate(self, userText, modelSpec=None, apiKey=None):
        return f"{modelSpec or 'fast'}: {userText}"

    def stream(self, userText, modelSpec=None, apiKey=None):
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
    prompts = []
    providers.setReplier(lambda prompt, spec, key: prompts.append((prompt, spec)) or "A short summary.")
    conversationId = ingestConversation("other", "sum", chat(("What are cats?", "Small mammals.")))["conversationId"]
    summary = buildLlmSummary(conversationId)
    assert summary["mode"] == "llm" and summary["summary"] == "A short summary."
    assert prompts[0][1] is None
    providers.setReplier(lambda prompt, spec, key: "A different summary.")
    assert buildLlmSummary(conversationId)["summary"] == "A different summary.", "a new replier must not get the old one's cached text"


# --- installing and restoring ----------------------------------------------

def test_setting_none_restores_the_builtin_providers():
    providers.setReplier(lambda *_args: "x")
    assert isinstance(providers.getReplier(), providers.CallableReplier)
    assert isinstance(providers.setReplier(None), providers.BuiltinReplier)
    assert providers.getReplier().generate("hi") == "Stub LLM response for: hi"


def test_something_that_is_not_a_replier_is_rejected():
    with pytest.raises(TypeError):
        providers.setReplier(42)


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
