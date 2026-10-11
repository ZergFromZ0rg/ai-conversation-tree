"""The services enforce their own rules and report failures with domain
exceptions — no HTTP involved. These are the guarantees a host other than the
FastAPI app (a CLI, another application) relies on."""

import httpx
import pytest

from conversation_tree import chatService
from conversation_tree.analysisService import buildConversationReport, buildConversationSummary, buildLlmSummary, searchConversations
from conversation_tree.chatService import (
    analyzeConversation,
    createConversationEdge,
    createConversationSession,
    createManualConceptLink,
    deleteConversationSession,
    generateAiText,
    getConversationSession,
    listConversationGraph,
    listConversationTurns,
    patchConversationEdge,
    processChatMessage,
    removeConceptLink,
    removeConversationEdge,
    setConversationSessionModel,
    streamAiText,
    streamChatMessage,
)
from conversation_tree.contextService import buildContextHealth
from conversation_tree.conversationErrors import ConversationTreeError, InvalidInputError, NotFoundError, ProviderError
from conversation_tree.ingestService import ingestConversation
from conftest import chat

MISSING = 9999


@pytest.fixture()
def conversationId(client):
    """A conversation with three turns; the `client` fixture wipes the database afterwards."""
    return ingestConversation(
        "other",
        "contract",
        chat(
            ("What are cats?", "Cats are small carnivorous mammals."),
            ("What is the capital of France?", "Paris."),
            ("What do cats eat?", "Meat; they are obligate carnivores."),
        ),
    )["conversationId"]


def test_exceptions_are_catchable_as_builtins_and_as_one_base():
    assert issubclass(NotFoundError, (ConversationTreeError, LookupError))
    assert issubclass(InvalidInputError, (ConversationTreeError, ValueError))
    assert issubclass(ProviderError, (ConversationTreeError, RuntimeError))


# --- a missing conversation is NotFoundError everywhere -------------------

@pytest.mark.parametrize(
    "call",
    [
        lambda: getConversationSession(MISSING),
        lambda: setConversationSessionModel(MISSING, "stub"),
        lambda: deleteConversationSession(MISSING),
        lambda: listConversationTurns(MISSING),
        lambda: listConversationGraph(MISSING),
        lambda: analyzeConversation(MISSING),
        lambda: processChatMessage(MISSING, "hello"),
        lambda: streamChatMessage(MISSING, "hello"),
        lambda: buildConversationSummary(MISSING),
        lambda: buildLlmSummary(MISSING),
        lambda: buildConversationReport(MISSING),
        lambda: buildContextHealth(MISSING),
        lambda: createConversationEdge(MISSING, 0, 1, "related", 0.5),
    ],
)
def test_missing_conversation_raises_not_found(client, call):
    with pytest.raises(NotFoundError, match="Conversation not found"):
        call()


def test_missing_edge_and_link_raise_not_found(client):
    with pytest.raises(NotFoundError, match="Edge not found"):
        patchConversationEdge(MISSING, label="related")
    with pytest.raises(NotFoundError, match="Edge not found"):
        removeConversationEdge(MISSING)
    with pytest.raises(NotFoundError, match="Concept link not found"):
        removeConceptLink(MISSING)
    with pytest.raises(NotFoundError, match="concepts were not found"):
        createManualConceptLink("nope-a", "nope-b", "related")


# --- model specs ----------------------------------------------------------

def test_malformed_model_spec_is_invalid_input_everywhere(client, conversationId):
    with pytest.raises(InvalidInputError, match="model must be"):
        createConversationSession(None, "bogus")
    with pytest.raises(InvalidInputError):
        setConversationSessionModel(conversationId, "x:y")
    with pytest.raises(InvalidInputError):
        processChatMessage(conversationId, "hello", "bogus")
    with pytest.raises(InvalidInputError):
        buildLlmSummary(conversationId, "nonsense")


def test_not_found_takes_precedence_over_a_bad_model(client):
    with pytest.raises(NotFoundError):
        processChatMessage(MISSING, "hello", "bogus")


def test_empty_user_text_is_invalid_input(client, conversationId):
    with pytest.raises(InvalidInputError, match="userText"):
        processChatMessage(conversationId, "")


# --- edges ----------------------------------------------------------------

def test_edge_rules_are_enforced_in_the_service(client, conversationId):
    def edge(**overrides):
        arguments = {"fromTurnId": 0, "toTurnId": 1, "label": "related", "confidence": 0.5, **overrides}
        return createConversationEdge(conversationId, **arguments)

    with pytest.raises(InvalidInputError, match="label"):
        edge(label="nope")
    with pytest.raises(InvalidInputError, match="confidence"):
        edge(confidence=1.5)
    with pytest.raises(InvalidInputError, match="differ"):
        edge(fromTurnId=1, toTurnId=1)
    with pytest.raises(InvalidInputError, match="earlier"):
        edge(fromTurnId=2, toTurnId=1)
    with pytest.raises(NotFoundError, match="Turns not found"):
        edge(toTurnId=7)

    created = edge()
    assert created["origin"] == "manual"
    with pytest.raises(InvalidInputError):
        patchConversationEdge(created["id"], confidence=-0.1)
    assert patchConversationEdge(created["id"], label="branch")["label"] == "branch"
    removeConversationEdge(created["id"])
    with pytest.raises(NotFoundError):
        removeConversationEdge(created["id"])


def test_concept_link_rules_are_enforced_in_the_service(client):
    with pytest.raises(InvalidInputError, match="kind"):
        createManualConceptLink("a", "b", "bogus")
    with pytest.raises(InvalidInputError, match="differ"):
        createManualConceptLink("a", "a", "same")


# --- search and ingest ----------------------------------------------------

def test_search_validates_its_own_arguments(client):
    with pytest.raises(InvalidInputError, match="query"):
        searchConversations("   ")
    with pytest.raises(InvalidInputError, match="limit"):
        searchConversations("cats", limit=0)
    with pytest.raises(InvalidInputError, match="limit"):
        searchConversations("cats", limit=101)


def test_ingest_validates_its_own_arguments(client):
    with pytest.raises(InvalidInputError, match="source"):
        ingestConversation("bing", "x", chat(("hi", "hello")))
    with pytest.raises(InvalidInputError, match="exchange"):
        ingestConversation("other", "x", [{"role": "user", "text": "unanswered"}])


# --- streaming validates before it streams --------------------------------

def test_stream_rejects_a_bad_request_at_call_time_not_on_iteration(client, conversationId):
    # Neither call is iterated: the error must already be raised.
    with pytest.raises(NotFoundError):
        streamChatMessage(MISSING, "hello")
    with pytest.raises(InvalidInputError):
        streamChatMessage(conversationId, "hello", "bogus")


def test_stream_yields_deltas_then_done(client, conversationId, monkeypatch):
    monkeypatch.setattr(chatService, "stubStreamDelaySeconds", 0)
    events = list(streamChatMessage(conversationId, "What do dogs eat?"))
    assert [event["type"] for event in events[:-1]] == ["delta"] * (len(events) - 1)
    assert events[-1]["type"] == "done"


# --- provider failures ----------------------------------------------------

def test_missing_api_key_is_a_provider_error(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        generateAiText("hi", "openai:gpt-test")


def test_transport_errors_become_provider_errors(client, monkeypatch):
    def unreachable(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(chatService, "generateOllamaText", unreachable)
    monkeypatch.setattr(chatService, "generateOllamaTextStream", unreachable)
    with pytest.raises(ProviderError, match="connection refused") as raised:
        generateAiText("hi", "ollama:tiny")
    assert isinstance(raised.value.__cause__, httpx.ConnectError)
    with pytest.raises(ProviderError, match="connection refused"):
        list(streamAiText("hi", "ollama:tiny"))
