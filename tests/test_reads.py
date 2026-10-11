"""Reads fetch only what they use: no embedding vectors for text, no other
conversations for a scoped query, and one read per conversation per call."""

import pytest

from conftest import chat
from conversation_tree import analysisService, chatService, db, ingestService, providers
from conversation_tree.analysisService import (
    buildConversationReport,
    buildConversationSummary,
    buildLlmSummary,
    searchConversations,
)
from conversation_tree.chatService import (
    listConceptGraph,
    listConversationConceptLinks,
    listConversationGraph,
    listConversationTurns,
    processChatMessage,
)
from conversation_tree.contextService import buildContextHealth
from conversation_tree.ingestService import ingestConversation

CATS = chat(("What are cats?", "Small mammals."), ("What do cats eat?", "Meat."))
DOGS = chat(("What are dogs?", "Domesticated canines."), ("What do dogs eat?", "Meat too."))


@pytest.fixture()
def two(client):
    return (
        ingestConversation("other", "cats", CATS)["conversationId"],
        ingestConversation("other", "dogs", DOGS)["conversationId"],
    )


@pytest.fixture()
def turnReads(monkeypatch):
    """Record whether each read of a conversation's turns asked for the embeddings."""
    asked = []
    real = db.getConversationTurns

    def spy(conversationId, withEmbeddings=True):
        asked.append(withEmbeddings)
        return real(conversationId, withEmbeddings=withEmbeddings)

    for module in (chatService, analysisService, ingestService, db):  # db: contextService imports it lazily
        monkeypatch.setattr(module, "getConversationTurns", spy)
    return asked


# --- the database functions -------------------------------------------------------

def test_turns_come_back_without_embeddings_on_request(two):
    cats, _dogs = two
    with_vectors = db.getConversationTurns(cats)
    text_only = db.getConversationTurns(cats, withEmbeddings=False)
    assert all(len(turn["embedding"]) == 384 * 4 for turn in with_vectors)
    assert all(turn["embedding"] is None for turn in text_only)
    strip = lambda turns: [{k: v for k, v in turn.items() if k != "embedding"} for turn in turns]
    assert strip(text_only) == strip(with_vectors), "everything but the vector is identical"


def test_concept_members_come_back_without_embeddings_on_request(two):
    with_vectors = db.getAllConceptMembers()
    text_only = db.getAllConceptMembers(withEmbeddings=False)
    assert with_vectors and all("embedding" in row for row in with_vectors)
    assert all("embedding" not in row for row in text_only)
    assert text_only == [{k: v for k, v in row.items() if k != "embedding"} for row in with_vectors]


def test_search_inputs_can_be_scoped_to_one_conversation(two):
    cats, dogs = two
    assert {row["conversationId"] for row in db.getAllSearchTurns()} == {cats, dogs}
    assert {row["conversationId"] for row in db.getAllSearchTurns(cats)} == {cats}
    assert {row["conversationId"] for row in db.getAllTurnEmbeddings()} == {cats, dogs}
    assert {row["conversationId"] for row in db.getAllTurnEmbeddings(dogs)} == {dogs}


# --- callers ask for no more than they use -------------------------------------------

def test_readers_that_only_need_text_never_ask_for_the_vectors(two, turnReads):
    cats, _dogs = two
    providers.setReplier(lambda request: "A summary.")
    listConversationTurns(cats)
    listConversationGraph(cats)
    buildContextHealth(cats)
    buildConversationSummary(cats)
    buildConversationReport(cats)
    buildLlmSummary(cats)
    processChatMessage(cats, "And mice?", "a-model")
    ingestConversation("other", "cats", CATS + chat(("Do cats purr?", "Yes.")))  # compares stored text
    assert len(turnReads) >= 8
    assert not any(turnReads), "a text-only reader asked for the embedding vectors"


def test_the_concept_views_do_not_load_every_embedding_in_the_workspace(two, monkeypatch):
    cats, _dogs = two
    asked = []
    real = db.getAllConceptMembers

    def spy(withEmbeddings=True):
        asked.append(withEmbeddings)
        return real(withEmbeddings=withEmbeddings)

    monkeypatch.setattr(chatService, "getAllConceptMembers", spy)
    listConceptGraph()
    listConversationConceptLinks(cats)
    assert asked == [False, False]


@pytest.mark.parametrize(
    "call",
    [
        lambda cid: buildConversationSummary(cid),
        lambda cid: buildConversationReport(cid),
        lambda cid: buildLlmSummary(cid),
    ],
    ids=["summary", "report", "llm summary"],
)
def test_each_analysis_call_reads_the_conversation_once(two, turnReads, call):
    cats, _dogs = two
    providers.setReplier(lambda request: "A summary.")
    turnReads.clear()
    call(cats)
    assert len(turnReads) == 1


# --- a scoped search stays scoped ---------------------------------------------------------

def test_a_search_scoped_to_one_chat_matches_the_workspace_search_filtered_to_it(two, monkeypatch):
    cats, _dogs = two
    scopes = {"turns": [], "vectors": []}
    realTurns, realVectors = db.getAllSearchTurns, db.getAllTurnEmbeddings

    def spyTurns(conversationId=None):
        scopes["turns"].append(conversationId)
        return realTurns(conversationId)

    def spyVectors(conversationId=None):
        scopes["vectors"].append(conversationId)
        return realVectors(conversationId)

    monkeypatch.setattr(analysisService, "getAllSearchTurns", spyTurns)
    monkeypatch.setattr(analysisService, "getAllTurnEmbeddings", spyVectors)

    everywhere = searchConversations("what do they eat", limit=50)["results"]
    assert scopes == {"turns": [None], "vectors": [None]}

    scoped = searchConversations("what do they eat", limit=50, conversationId=cats)["results"]
    assert scopes == {"turns": [None, cats], "vectors": [None, cats]}, "the second search read only that chat"

    assert scoped and {hit["conversationId"] for hit in scoped} == {cats}
    assert scoped == [hit for hit in everywhere if hit["conversationId"] == cats]


def test_a_scoped_search_asks_the_vector_index_about_one_chat_only(two, monkeypatch):
    cats, _dogs = two

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a scoped search must not query the whole workspace index")

    monkeypatch.setattr(analysisService, "searchWorkspace", forbidden)
    monkeypatch.setattr(analysisService, "searchConversation", lambda conversationId, embedding, exclude, limit: [(0, 0.95)])
    hits = searchConversations("cats", conversationId=cats)["results"]
    assert {hit["conversationId"] for hit in hits} == {cats}
    assert [hit["semanticScore"] for hit in hits if hit["turnId"] == 0] == [0.95]
