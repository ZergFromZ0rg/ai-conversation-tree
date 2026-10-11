from datetime import datetime
from typing import NamedTuple

from .conversationErrors import InvalidInputError, NotFoundError, ProviderError
from .db import (
    applyReclassification,
    createEdge,
    createConversation,
    createManualConceptLink as createManualConceptLinkDb,
    deleteConceptLink,
    deleteConversation,
    deleteEdge,
    getAllConceptMembers,
    getConceptMembership,
    getConversation,
    getConversationTurnIds,
    getConversationTurns,
    getEdge,
    listAllConceptLinks,
    listConceptLinksForConversation,
    listConversations,
    listConversationEdges,
    saveTurnBundle,
    setConversationModel,
    setConversationTitle,
    updateEdge,
)
from .conceptIndex import (
    conceptLabelsFromMembers,
    forgetConversation,
    matchConceptKeys,
    maybeRelinkConcepts,
)
from .graphService import addTurn, reclassifyTurns
from .graphStore import invalidate, lockedGraph
from .contextService import recentHistory
from .providers import ReplyRequest, contextWindowFor, getReplier, requireValidModelSpec

EDGE_LABELS = ("continuation", "branch", "related")
CONCEPT_LINK_KINDS = ("same", "related")




def serializeGraphNode(nodeId: int, userText: str, aiText: str, conceptIds: list[int], root: bool, timelineParent: int | None) -> dict:
    return {
        "id": int(nodeId),
        "userText": str(userText),
        "aiText": str(aiText),
        "conceptIds": [int(conceptId) for conceptId in conceptIds],
        "root": bool(root),
        "timelineParent": timelineParent,
    }


def serializeGraphEdge(edgeId: int | None, fromTurnId: int, toTurnId: int, label: str, confidence: float, origin: str = "auto") -> dict:
    return {
        "id": int(edgeId) if edgeId is not None else None,
        "fromTurnId": int(fromTurnId),
        "toTurnId": int(toTurnId),
        "label": str(label),
        "confidence": float(confidence),
        "origin": str(origin),
    }


def buildGraphPayloadFromStoredTurns(conversationId: int, storedTurns: list[dict]) -> dict:
    nodes = [
        serializeGraphNode(
            nodeId=turn["id"],
            userText=turn["userText"],
            aiText=turn["aiText"],
            conceptIds=turn["conceptIds"],
            root=turn["root"],
            timelineParent=turn["timelineParent"],
        )
        for turn in sorted(storedTurns, key=lambda turn: int(turn["id"]))
    ]

    edges = [
        serializeGraphEdge(
            edgeId=edge["id"],
            fromTurnId=edge["fromTurnId"],
            toTurnId=edge["toTurnId"],
            label=edge["label"],
            confidence=edge["confidence"],
            origin=edge.get("origin", "auto"),
        )
        for edge in sorted(
            listConversationEdges(conversationId),
            key=lambda edge: (int(edge["toTurnId"]), int(edge["fromTurnId"]), int(edge["id"]))
        )
    ]

    return {
        "nodes": nodes,
        "edges": edges,
    }




def serializeStoredTurns(storedTurns: list[dict]) -> list[dict]:
    return [
        {
            "id": turn["id"],
            "conversationId": turn["conversationId"],
            "userText": turn["userText"],
            "aiText": turn["aiText"],
            "timestamp": turn["timestamp"],
            "root": turn["root"],
            "timelineParent": turn["timelineParent"],
            "conceptIds": turn["conceptIds"],
            "semanticParents": turn["semanticParents"],
        }
        for turn in storedTurns
    ]


def buildConversationPayload(conversationId: int) -> dict:
    storedTurns = getConversationTurns(conversationId, withEmbeddings=False)
    graph = buildGraphPayloadFromStoredTurns(conversationId, storedTurns)
    return {
        "conversationId": conversationId,
        "turns": serializeStoredTurns(storedTurns),
        "nodes": graph["nodes"],
        "edges": graph["edges"],
    }


def turnBundle(turn) -> dict:
    """A classified turn as the keyword arguments the database stores it with."""
    return dict(
        turnId=turn.id,
        userText=turn.userText,
        aiText=turn.aiText,
        timestamp=turn.timestamp.isoformat() if isinstance(turn.timestamp, datetime) else str(turn.timestamp),
        root=turn.root,
        timelineParent=turn.timelineParent,
        embeddingBytes=turn.embedding.astype("float32").tobytes(),
        semanticParents=turn.semanticParents,
        conceptIds=turn.conceptIds,
    )


def persistGraphTurn(conversationId: int, turn) -> None:
    saveTurnBundle(conversationId=conversationId, **turnBundle(turn))




def _requireConversation(conversationId: int) -> dict:
    conversation = getConversation(conversationId)
    if conversation is None:
        raise NotFoundError("Conversation not found.")
    return conversation


def createConversationSession(title: str | None = None, model: str | None = None) -> dict:
    requireValidModelSpec(model)
    conversationId = createConversation(title)
    if model:
        setConversationModel(conversationId, model)
    return {"conversationId": conversationId}


def listConversationSessions() -> dict:
    return {"conversations": listConversations()}


def getConversationSession(conversationId: int) -> dict:
    return _requireConversation(conversationId)


def setConversationSessionModel(conversationId: int, model: str | None) -> dict:
    requireValidModelSpec(model)
    conversation = setConversationModel(conversationId, model)
    if conversation is None:
        raise NotFoundError("Conversation not found.")
    return conversation


def loadConversationSession(conversationId: int) -> dict:
    # Touch the cache so the graph is warm for the next turn.
    with lockedGraph(conversationId):
        return buildConversationPayload(conversationId)


def deleteConversationSession(conversationId: int) -> None:
    deleted = deleteConversation(conversationId)
    invalidate(conversationId)
    forgetConversation(conversationId)
    if not deleted:
        raise NotFoundError("Conversation not found.")


def listConversationTurns(conversationId: int) -> dict:
    _requireConversation(conversationId)
    return {
        "conversationId": conversationId,
        "turns": serializeStoredTurns(getConversationTurns(conversationId, withEmbeddings=False)),
    }


def listConversationGraph(conversationId: int) -> dict:
    _requireConversation(conversationId)
    storedTurns = getConversationTurns(conversationId, withEmbeddings=False)
    graph = buildGraphPayloadFromStoredTurns(conversationId, storedTurns)
    return {"conversationId": conversationId, "nodes": graph["nodes"], "edges": graph["edges"]}


def _conceptTurnCounts(members: list[dict]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], set[int]] = {}
    for row in members:
        counts.setdefault((row["conversationId"], row["conceptId"]), set()).add(row["turnId"])
    return {key: len(turnIds) for key, turnIds in counts.items()}


def _serializeConceptEdge(link: dict) -> dict:
    return {
        "id": link["id"],
        "a": {
            "conversationId": link["aConversationId"],
            "conceptId": link["aConceptId"],
            "conceptKey": link["aConceptKey"],
        },
        "b": {
            "conversationId": link["bConversationId"],
            "conceptId": link["bConceptId"],
            "conceptKey": link["bConceptKey"],
        },
        "score": link["score"],
        "kind": link["label"],  # 'same' | 'related'
        "origin": link["origin"],  # 'auto' | 'manual'
    }


def listConceptGraph() -> dict:
    """Workspace-wide concept graph: every concept as a node, links as edges."""
    members = getAllConceptMembers(withEmbeddings=False)  # labels and counts only read text
    labels = conceptLabelsFromMembers(members)
    turnCounts = _conceptTurnCounts(members)
    conceptKeys = {
        (row["conversationId"], row["conceptId"]): row["conceptKey"] for row in members
    }
    titles = {conversation["id"]: conversation["title"] for conversation in listConversations()}

    nodes = [
        {
            "conversationId": conversationId,
            "conceptId": conceptId,
            "conceptKey": conceptKeys.get((conversationId, conceptId)),
            "label": label,
            "turnCount": turnCounts.get((conversationId, conceptId), 0),
            "conversationTitle": titles.get(conversationId),
        }
        for (conversationId, conceptId), label in sorted(labels.items())
    ]
    edges = [_serializeConceptEdge(link) for link in listAllConceptLinks()]
    return {"nodes": nodes, "edges": edges}


def listConversationConceptLinks(conversationId: int) -> dict:
    """This conversation's concepts and, per concept, where else each is discussed.

    Shaped for the transcript drawer's "also discussed in" affordance: grouped
    by the local conceptId, each entry naming the far conversation and concept.
    """
    _requireConversation(conversationId)
    members = getAllConceptMembers(withEmbeddings=False)  # labels and counts only read text
    labels = conceptLabelsFromMembers(members)
    titles = {conversation["id"]: conversation["title"] for conversation in listConversations()}

    groups: dict[int, list[dict]] = {}
    for link in listConceptLinksForConversation(conversationId):
        endpointA = (link["aConversationId"], link["aConceptId"])
        endpointB = (link["bConversationId"], link["bConceptId"])
        if endpointA[0] == conversationId:
            near, far, farKey = endpointA, endpointB, link["bConceptKey"]
        else:
            near, far, farKey = endpointB, endpointA, link["aConceptKey"]
        groups.setdefault(near[1], []).append(
            {
                "linkId": link["id"],
                "conversationId": far[0],
                "conceptId": far[1],
                "conceptKey": farKey,
                "conversationTitle": titles.get(far[0]),
                "label": labels.get(far),
                "score": link["score"],
                "kind": link["label"],
                "origin": link["origin"],
            }
        )

    concepts = [
        {
            "conceptId": conceptId,
            "label": labels.get((conversationId, conceptId)),
            "links": sorted(entries, key=lambda entry: entry["score"], reverse=True),
        }
        for conceptId, entries in sorted(groups.items())
    ]
    return {"conversationId": conversationId, "concepts": concepts}


def createManualConceptLink(conceptKeyA: str, conceptKeyB: str, kind: str) -> dict:
    """Hand-link two concepts. Manual links carry score 1.0 (human-asserted) and
    survive re-analysis of either side as long as both concepts still exist —
    see conceptKey in the persistence model. Raises InvalidInputError for an
    unknown kind or a concept linked to itself, NotFoundError if either key
    is unknown."""
    if kind not in CONCEPT_LINK_KINDS:
        raise InvalidInputError(f"kind must be one of {list(CONCEPT_LINK_KINDS)}.")
    if conceptKeyA == conceptKeyB:
        raise InvalidInputError("aConceptKey and bConceptKey must differ.")
    link = createManualConceptLinkDb(conceptKeyA, conceptKeyB, 1.0, kind)
    if link is None:
        raise NotFoundError("One or both concepts were not found.")
    return _serializeConceptEdge(link)


def removeConceptLink(linkId: int) -> None:
    if not deleteConceptLink(linkId):
        raise NotFoundError("Concept link not found.")


conversationTitleMaxChars = 60


def titleFromText(text: str) -> str:
    """First user message, collapsed and trimmed to a word boundary."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= conversationTitleMaxChars:
        return collapsed
    head = collapsed[:conversationTitleMaxChars].rsplit(" ", 1)[0].rstrip()
    return (head or collapsed[:conversationTitleMaxChars].rstrip()) + "…"


def _requireText(value, name: str) -> None:
    """A message must contain something other than whitespace: a blank one would
    be classified, stored, and (as a first turn) give the conversation an empty
    title that nothing later replaces."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidInputError(f"{name} must not be empty.")


class _TurnPlan(NamedTuple):
    modelSpec: str | None
    """The model to ask for; None means the replier's own default."""

    rememberModel: str | None
    """An explicit model that should become the conversation's default, once the turn has succeeded."""


def _prepareTurn(conversationId: int, userText: str, model: str | None) -> _TurnPlan:
    """Everything that can reject a turn, checked before any model call or
    write: the text is non-empty, the conversation exists (NotFoundError, which
    takes precedence), and the model spec is well-formed.

    Then pick the model: an explicit one on the request wins, otherwise the
    conversation's stored choice, otherwise the replier's own default. An
    explicit model becomes the conversation's default only after the turn has
    succeeded (see ``rememberModel``): a request that fails must leave the
    conversation as it found it."""
    _requireText(userText, "userText")
    conversation = _requireConversation(conversationId)
    requireValidModelSpec(model)
    remember = model if model and conversation.get("model") != model else None
    return _TurnPlan(model or conversation.get("model"), remember)


def appendTurn(conversationId: int, userText: str, aiText: str) -> dict:
    """Record a finished exchange in a conversation.

    The entry point for a host that produces replies itself: classify the turn
    against the conversation's earlier ones, persist it, title the conversation
    from its first question, and refresh cross-chat concept links. Returns the
    payload ``processChatMessage`` returns (``turnId``, ``aiText``, ``turns``,
    ``nodes``, ``edges``). Raises NotFoundError for an unknown conversation and
    InvalidInputError for empty text.
    """
    _requireText(userText, "userText")
    _requireText(aiText, "aiText")
    _requireConversation(conversationId)
    return _finalizeTurn(conversationId, userText, aiText)


def _finalizeTurn(conversationId: int, userText: str, aiText: str, rememberModel: str | None = None) -> dict:
    """Shared tail of the blocking and streaming send paths: classify and
    persist the turn, auto-title the conversation on its first turn, and
    refresh cross-chat concept links. Assumes the AI reply is already in hand.
    `rememberModel`, when given, becomes the conversation's default model."""
    conversation = getConversation(conversationId)

    with lockedGraph(conversationId) as graph:
        turn = addTurn(graph, userText, aiText)
        persistGraphTurn(conversationId, turn)
        payload = buildConversationPayload(conversationId)

    # This conversation's very first turn: title it from the question, unless
    # it was already given a title (e.g. POST /conversations {"title": ...}).
    if turn.timelineParent is None and not (conversation or {}).get("title"):
        setConversationTitle(conversationId, titleFromText(userText))

    if rememberModel:
        setConversationModel(conversationId, rememberModel)

    # Outside the lock: refresh this conversation's cross-chat concept links
    # (debounced, best-effort — never fails the turn).
    maybeRelinkConcepts(conversationId)

    payload["turnId"] = turn.id
    payload["aiText"] = turn.aiText
    return payload


def _requireReplyText(aiText) -> str:
    """A reply must be real text. The built-in providers already refuse to return
    nothing; a plugged-in replier gets the same rule here, so a blank or
    non-text reply is a provider failure instead of an empty turn that would be
    stored and then sent back to the model as an empty message."""
    if not isinstance(aiText, str) or not aiText.strip():
        raise ProviderError("Model returned no text output.")
    return aiText


def _replyRequest(conversationId: int, userText: str, modelSpec: str | None, apiKey: str | None) -> ReplyRequest:
    """The request for a reply to `userText`: the model, plus as much of the
    conversation so far as fits that model's context window."""
    history = recentHistory(
        getConversationTurns(conversationId, withEmbeddings=False), userText, contextWindowFor(modelSpec)
    )
    return ReplyRequest(userText, modelSpec, apiKey, conversationId, history)


def processChatMessage(
    conversationId: int, userText: str, model: str | None = None, apiKey: str | None = None
) -> dict:
    plan = _prepareTurn(conversationId, userText, model)
    # Generate the AI reply before taking the lock so the (potentially slow) LLM
    # call does not serialize other work on this conversation's graph.
    aiText = _requireReplyText(
        getReplier().generate(_replyRequest(conversationId, userText, plan.modelSpec, apiKey))
    )
    return _finalizeTurn(conversationId, userText, aiText, plan.rememberModel)


def streamChatMessage(conversationId: int, userText: str, model: str | None = None, apiKey: str | None = None):
    """Iterator of SSE-ready event dicts for one turn.

    Yields any number of {"type": "delta", "text": "..."} as the reply comes
    in, then exactly one {"type": "done", ...} carrying the same payload shape
    processChatMessage returns (turnId, aiText, turns, nodes, edges). Persists
    nothing until the reply is complete — same ordering as the blocking path,
    just with the network wait broken into visible pieces.

    Request validation (NotFoundError / InvalidInputError) happens when this
    is *called*, not on first iteration, so a caller can still turn a bad
    request into an error response before it commits to streaming. Provider
    failures (ProviderError) surface while iterating.
    """
    plan = _prepareTurn(conversationId, userText, model)
    request = _replyRequest(conversationId, userText, plan.modelSpec, apiKey)
    return _streamTurnEvents(conversationId, userText, request, plan.rememberModel)


def _streamTurnEvents(conversationId: int, userText: str, request: ReplyRequest, rememberModel: str | None):
    chunks: list[str] = []
    for chunk in getReplier().stream(request):
        if not isinstance(chunk, str):
            raise ProviderError("Model returned no text output.")
        chunks.append(chunk)
        yield {"type": "delta", "text": chunk}

    aiText = _requireReplyText("".join(chunks))

    payload = _finalizeTurn(conversationId, userText, aiText, rememberModel)
    yield {"type": "done", **payload}


def analyzeConversation(conversationId: int) -> dict:
    _requireConversation(conversationId)
    with lockedGraph(conversationId) as graph:
        oldMembership, oldKeys = getConceptMembership(conversationId)

        rebuiltTurns = reclassifyTurns(graph)

        newMembership: dict[int, set[int]] = {}
        for turn in rebuiltTurns:
            for conceptId in turn.conceptIds:
                newMembership.setdefault(conceptId, set()).add(turn.id)
        conceptKeyByConceptId = matchConceptKeys(oldMembership, oldKeys, newMembership)

        applyReclassification(
            conversationId,
            [
                (parentId, turn.id, label, confidence)
                for turn in rebuiltTurns
                for parentId, label, confidence in turn.semanticParents
            ],
            [
                (turn.id, turn.root, turn.timelineParent, turn.conceptIds)
                for turn in rebuiltTurns
            ],
            conceptKeyByConceptId,
        )

        payload = buildConversationPayload(conversationId)

    # conceptIds were just reassigned from zero, so every 'auto' link touching
    # this conversation is stale — force a rebuild.
    maybeRelinkConcepts(conversationId, force=True)

    return {
        "conversationId": conversationId,
        "turnCount": len(payload["turns"]),
        "edgeCount": len(payload["edges"]),
        "graph": {"nodes": payload["nodes"], "edges": payload["edges"]},
    }


def _requireValidEdgeFields(label: str | None, confidence: float | None) -> None:
    """Each field is checked only when given, so PATCH can pass a partial update."""
    if label is not None and label not in EDGE_LABELS:
        raise InvalidInputError(f"label must be one of {list(EDGE_LABELS)}.")
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise InvalidInputError("confidence must be between 0.0 and 1.0.")


def createConversationEdge(conversationId: int, fromTurnId: int, toTurnId: int, label: str, confidence: float) -> dict:
    """Hand-draw an edge from an earlier turn to a later one. Raises
    NotFoundError if the conversation or either turn doesn't exist,
    InvalidInputError for a bad label/confidence or a backwards/self edge."""
    _requireConversation(conversationId)
    _requireValidEdgeFields(label, confidence)
    if fromTurnId == toTurnId:
        raise InvalidInputError("fromTurnId and toTurnId must differ.")
    if fromTurnId > toTurnId:
        raise InvalidInputError(
            "fromTurnId must be earlier than toTurnId; edges point from the earlier turn to the later turn."
        )
    turnIds = set(getConversationTurnIds(conversationId))
    missing = [turnId for turnId in (fromTurnId, toTurnId) if turnId not in turnIds]
    if missing:
        raise NotFoundError(f"Turns not found in conversation: {missing}.")
    edge = createEdge(conversationId, fromTurnId, toTurnId, label, confidence)
    invalidate(conversationId)
    return edge


def patchConversationEdge(edgeId: int, label: str | None = None, confidence: float | None = None) -> dict:
    _requireValidEdgeFields(label, confidence)
    edge = updateEdge(edgeId, label=label, confidence=confidence)
    if edge is None:
        raise NotFoundError("Edge not found.")
    invalidate(edge["conversationId"])
    return edge


def removeConversationEdge(edgeId: int) -> None:
    edge = getEdge(edgeId)
    deleted = deleteEdge(edgeId)
    if deleted and edge is not None:
        invalidate(edge["conversationId"])
    if not deleted:
        raise NotFoundError("Edge not found.")
