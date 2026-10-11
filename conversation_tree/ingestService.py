"""Import conversations captured elsewhere (exports, scripts, other apps).

A captured chat arrives as a flat list of role-tagged messages. It is paired
into turns, matched to an earlier import of the same chat by
(source, externalId), and only the new turns are classified and stored — so
re-syncing a chat that grew by a few messages costs a few turns, not a rebuild.
"""

from datetime import datetime

from .chatService import titleFromText, persistGraphTurn
from .conceptIndex import maybeRelinkConcepts
from .conversationErrors import ConflictError, InvalidInputError
from .db import (
    clearConversationTurns,
    createConversation,
    findConversationBySource,
    getConversationTurns,
    setConversationTitle,
    updateConversationSource,
)
from .graphService import addTurn, resetConversation
from .graphStore import invalidate, lockedGraph

SOURCES = ("chatgpt", "claude", "gemini", "other")

_userRoles = frozenset({"user", "human", "you"})
_assistantRoles = frozenset({"assistant", "ai", "model", "bot", "chatgpt", "claude", "gemini"})


def _normalize(text: str) -> str:
    return " ".join(text.split())


def pairMessagesIntoTurns(messages: list[dict]) -> list[tuple[str, str]]:
    """Fold role-tagged messages into (userText, aiText) turns.

    Consecutive user messages merge into one prompt, consecutive assistant
    messages into one reply (some UIs render a long answer as several blocks).
    An assistant message with no user message before it (a greeting) and a
    trailing user message with no reply yet (the answer is still being
    generated) are dropped — the next sync picks the finished turn up.
    """
    turns: list[tuple[str, str]] = []
    pendingUser: list[str] = []
    pendingAi: list[str] = []

    def flush() -> None:
        if pendingUser and pendingAi:
            turns.append(("\n\n".join(pendingUser), "\n\n".join(pendingAi)))
        pendingUser.clear()
        pendingAi.clear()

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        text = str(message.get("text", "")).strip()
        if not text:
            continue
        if role in _userRoles:
            if pendingAi:
                flush()
            pendingUser.append(text)
        elif role in _assistantRoles:
            if pendingUser:
                pendingAi.append(text)
            # else: leading assistant message with nothing to answer — drop.
    flush()
    return turns


def _sharedPrefixLength(stored: list[tuple[str, str]], incoming: list[tuple[str, str]]) -> int:
    shared = 0
    for (storedUser, storedAi), (newUser, newAi) in zip(stored, incoming):
        if _normalize(storedUser) != _normalize(newUser) or _normalize(storedAi) != _normalize(newAi):
            break
        shared += 1
    return shared


def ingestConversation(
    source: str,
    externalId: str,
    messages: list[dict],
    title: str | None = None,
    url: str | None = None,
) -> dict:
    """Create or update the conversation captured from `source`/`externalId`.

    Returns {conversationId, created, addedTurns, rebuilt, turnCount}.
    `rebuilt` is True when the stored history disagreed with the incoming one
    (an edited prompt or regenerated answer) and was replaced wholesale.

    Raises InvalidInputError for an unknown `source` or when the messages
    contain no complete user/assistant exchange.
    """
    if source not in SOURCES:
        raise InvalidInputError(f"source must be one of {list(SOURCES)}.")
    incoming = pairMessagesIntoTurns(messages)
    if not incoming:
        raise InvalidInputError("No complete user/assistant exchange found in the captured messages.")

    cleanTitle = _normalize(title)[:200] if title else None
    existing = findConversationBySource(source, externalId)
    created = existing is None
    if created:
        try:
            conversationId = createConversation(cleanTitle, source, externalId, url)
        except ConflictError:
            # A concurrent sync of the same new chat (auto-sync racing the
            # popup button) created it first; join that conversation.
            existing = findConversationBySource(source, externalId)
            created = False
            conversationId = existing["id"]
    else:
        conversationId = existing["id"]
    if not created:
        updateConversationSource(conversationId, cleanTitle, url)

    rebuilt = False
    with lockedGraph(conversationId) as graph:
        stored = [(turn["userText"], turn["aiText"]) for turn in getConversationTurns(conversationId)]
        shared = _sharedPrefixLength(stored, incoming)
        if shared < len(stored):
            # Diverged: drop the old classification and start over.
            clearConversationTurns(conversationId)
            resetConversation(graph)
            shared = 0
            rebuilt = True

        for userText, aiText in incoming[shared:]:
            turn = addTurn(graph, userText, aiText)
            turn.timestamp = datetime.now()
            persistGraphTurn(conversationId, turn)
        addedTurns = len(incoming) - shared
    invalidate(conversationId)

    if created and not cleanTitle:
        setConversationTitle(conversationId, titleFromText(incoming[0][0]))

    if addedTurns:
        maybeRelinkConcepts(conversationId, force=True)

    return {
        "conversationId": conversationId,
        "created": created,
        "addedTurns": addedTurns,
        "rebuilt": rebuilt,
        "turnCount": len(incoming),
    }
