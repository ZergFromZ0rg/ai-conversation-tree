"""appendTurn: a host that already has the reply records it without any model call."""

import pytest

from conversation_tree import db, providers
from conversation_tree.chatService import appendTurn, createConversationSession, getConversationSession
from conversation_tree.conversationErrors import ConflictError, InvalidInputError, NotFoundError


@pytest.fixture()
def noModel():
    """Fail loudly if anything tries to generate a reply."""

    def forbidden(*_args):
        raise AssertionError("appendTurn must not call the replier")

    providers.setReplier(forbidden)


@pytest.fixture()
def conversationId(client, noModel):
    return createConversationSession()["conversationId"]


def test_records_the_exchange_and_returns_the_turn_payload(conversationId):
    first = appendTurn(conversationId, "What are cats?", "Small carnivorous mammals.")
    assert first["turnId"] == 0
    assert first["aiText"] == "Small carnivorous mammals."
    assert [turn["userText"] for turn in first["turns"]] == ["What are cats?"]
    assert len(first["nodes"]) == 1

    second = appendTurn(conversationId, "What do cats eat?", "Meat.")
    assert second["turnId"] == 1
    assert len(second["turns"]) == 2
    assert any(edge["toTurnId"] == 1 for edge in second["edges"]), "the new turn is linked to an earlier one"


def test_titles_the_conversation_from_its_first_question_only(conversationId):
    appendTurn(conversationId, "What are cats?", "Small mammals.")
    appendTurn(conversationId, "What do cats eat?", "Meat.")
    assert getConversationSession(conversationId)["title"] == "What are cats?"


def test_keeps_a_title_the_caller_already_gave(client, noModel):
    conversationId = createConversationSession("My notes")["conversationId"]
    appendTurn(conversationId, "What are cats?", "Small mammals.")
    assert getConversationSession(conversationId)["title"] == "My notes"


def test_rejects_bad_input(conversationId):
    with pytest.raises(InvalidInputError, match="userText"):
        appendTurn(conversationId, "", "reply")
    with pytest.raises(InvalidInputError, match="aiText"):
        appendTurn(conversationId, "question", "")
    with pytest.raises(NotFoundError):
        appendTurn(9999, "question", "reply")


def test_a_failed_append_leaves_no_turn_behind(conversationId):
    with pytest.raises(InvalidInputError):
        appendTurn(conversationId, "question", "")
    assert db.getConversationTurns(conversationId) == []


def test_creating_a_duplicate_import_is_a_conflict_not_a_storage_error(client):
    db.createConversation("t", "other", "same-id", None)
    with pytest.raises(ConflictError, match="same-id") as raised:
        db.createConversation("t", "other", "same-id", None)
    assert raised.value.__cause__ is not None
    db.createConversation("t", None, None, None)  # conversations with no source never collide
    db.createConversation("t", None, None, None)
