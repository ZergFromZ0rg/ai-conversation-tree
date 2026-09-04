from datetime import datetime
import json
import os
import time
import httpx

from db import (
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
    getConversationTurns,
    getEdge,
    listAllConceptLinks,
    listConceptLinksForConversation,
    listConversations,
    listConversationEdges,
    saveConceptIds,
    saveSemanticEdges,
    saveTurn,
    saveTurnEmbedding,
    setConversationModel,
    setConversationTitle,
    updateEdge,
)
from conceptIndex import (
    conceptLabelsFromMembers,
    forgetConversation,
    matchConceptKeys,
    maybeRelinkConcepts,
)
from graphService import addTurn, reclassifyTurns
from graphStore import invalidate, lockedGraph


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


def extractResponseText(responseBody: dict) -> str:
    outputItems = responseBody.get("output", [])
    textParts: list[str] = []

    for outputItem in outputItems:
        for contentItem in outputItem.get("content", []):
            if contentItem.get("type") == "output_text":
                textParts.append(contentItem.get("text", ""))

    return "\n".join(part for part in textParts if part).strip()


def extractOllamaText(responseBody: dict) -> str:
    message = responseBody.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    return ""


KNOWN_PROVIDERS = ("ollama", "openai", "stub")


def parseModelSpec(modelSpec: str | None) -> tuple[str | None, str | None]:
    """Split "provider:model" into (provider, model).

    Ollama tags contain colons ("llama3:8b"), so only the first colon is a
    separator. "stub" and a bare provider yield model=None. A falsy spec yields
    (None, None) -> fall back to the environment-configured provider.
    """
    if not modelSpec:
        return None, None
    provider, _, model = modelSpec.partition(":")
    return provider, (model or None)


def ollamaBaseUrl() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def generateOllamaText(userText: str, model: str) -> str:
    response = httpx.post(
        f"{ollamaBaseUrl()}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": userText}],
            "stream": False,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    aiText = extractOllamaText(response.json())
    if not aiText:
        raise RuntimeError("Ollama returned no text output.")
    return aiText


def generateOpenAiText(userText: str, model: str) -> str:
    apiKey = os.environ.get("OPENAI_API_KEY")
    if not apiKey:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {apiKey}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": userText}],
                }
            ],
        },
        timeout=60.0,
    )
    response.raise_for_status()
    aiText = extractResponseText(response.json())
    if not aiText:
        raise RuntimeError("LLM returned no text output.")
    return aiText


def envDefaultModelSpec() -> str:
    """The model spec that matches the current environment-based routing."""
    ollamaModelName = os.environ.get("OLLAMA_MODEL")
    if ollamaModelName:
        return f"ollama:{ollamaModelName}"
    if os.environ.get("AI_CONVERSATION_TREE_STUB_LLM", "1") == "1":
        return "stub"
    if os.environ.get("OPENAI_API_KEY"):
        return f"openai:{os.environ.get('OPENAI_MODEL', 'gpt-5-mini')}"
    return "stub"


def generateAiText(userText: str, modelSpec: str | None = None) -> str:
    provider, model = parseModelSpec(modelSpec or envDefaultModelSpec())

    if provider == "stub":
        return f"Stub LLM response for: {userText}"
    if provider == "ollama":
        if not model:
            raise RuntimeError("An ollama model spec needs a model name, e.g. 'ollama:llama3'.")
        return generateOllamaText(userText, model)
    if provider == "openai":
        return generateOpenAiText(userText, model or os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    raise RuntimeError(f"Unknown model provider: {provider!r}. Expected one of {list(KNOWN_PROVIDERS)}.")


# ---- streaming ----
# Splits a reply into incremental text chunks as it's generated, instead of
# waiting for the whole thing. Classification still needs the complete text
# (it embeds userText + " " + aiText), so streaming only changes what the
# composer shows while the reply is in flight — persistence happens the same
# way afterward, once the stream ends (see streamChatMessage).

stubStreamWordsPerChunk = 3
stubStreamDelaySeconds = 0.05


def _streamStubText(userText: str):
    words = f"Stub LLM response for: {userText}".split(" ")
    for index in range(0, len(words), stubStreamWordsPerChunk):
        chunk = " ".join(words[index : index + stubStreamWordsPerChunk])
        if index > 0:
            chunk = " " + chunk
        time.sleep(stubStreamDelaySeconds)  # otherwise a local stub reply streams instantly
        yield chunk


def generateOllamaTextStream(userText: str, model: str):
    with httpx.stream(
        "POST",
        f"{ollamaBaseUrl()}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": userText}],
            "stream": True,
        },
        timeout=60.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            content = (chunk.get("message") or {}).get("content", "")
            if content:
                yield content
            if chunk.get("done"):
                break


def streamAiText(userText: str, modelSpec: str | None = None):
    """Yield a reply as it's generated. Same provider routing as generateAiText.

    OpenAI's Responses API streaming protocol isn't implemented (nothing here
    can exercise it without a live key) — that path falls back to the blocking
    call and yields the whole reply as one chunk, so the streaming endpoint
    still works end to end for it, just without token-by-token output.
    """
    provider, model = parseModelSpec(modelSpec or envDefaultModelSpec())

    if provider == "stub":
        yield from _streamStubText(userText)
        return
    if provider == "ollama":
        if not model:
            raise RuntimeError("An ollama model spec needs a model name, e.g. 'ollama:llama3'.")
        yield from generateOllamaTextStream(userText, model)
        return
    if provider == "openai":
        yield generateOpenAiText(userText, model or os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
        return
    raise RuntimeError(f"Unknown model provider: {provider!r}. Expected one of {list(KNOWN_PROVIDERS)}.")


def listAvailableModels() -> dict:
    options: list[dict] = []

    ollamaReachable = False
    try:
        response = httpx.get(f"{ollamaBaseUrl()}/api/tags", timeout=2.0)
        response.raise_for_status()
        ollamaReachable = True
        for entry in sorted(response.json().get("models", []), key=lambda m: m.get("name", "")):
            name = entry.get("name")
            if name:
                options.append({"id": f"ollama:{name}", "label": name, "provider": "ollama"})
    except Exception:
        pass

    if os.environ.get("OPENAI_API_KEY"):
        openAiModel = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        options.append(
            {"id": f"openai:{openAiModel}", "label": f"OpenAI · {openAiModel}", "provider": "openai"}
        )

    stubEnabled = os.environ.get("AI_CONVERSATION_TREE_STUB_LLM", "1") == "1"
    if stubEnabled or not options:
        options.append({"id": "stub", "label": "Stub (canned replies)", "provider": "stub"})

    return {
        "models": options,
        "default": envDefaultModelSpec(),
        "ollamaReachable": ollamaReachable,
    }


def isValidModelSpec(modelSpec: str) -> bool:
    provider, model = parseModelSpec(modelSpec)
    if provider not in KNOWN_PROVIDERS:
        return False
    if provider in ("ollama", "openai") and not model:
        return False
    return True


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
    storedTurns = getConversationTurns(conversationId)
    graph = buildGraphPayloadFromStoredTurns(conversationId, storedTurns)
    return {
        "conversationId": conversationId,
        "turns": serializeStoredTurns(storedTurns),
        "nodes": graph["nodes"],
        "edges": graph["edges"],
    }


def persistGraphTurn(conversationId: int, turn) -> None:
    saveTurn(
        conversationId=conversationId,
        turnId=turn.id,
        userText=turn.userText,
        aiText=turn.aiText,
        timestamp=turn.timestamp.isoformat() if isinstance(turn.timestamp, datetime) else str(turn.timestamp),
        root=turn.root,
        timelineParent=turn.timelineParent,
    )
    saveTurnEmbedding(conversationId, turn.id, turn.embedding.astype("float32").tobytes())
    saveSemanticEdges(conversationId, turn.id, turn.semanticParents)
    saveConceptIds(conversationId, turn.id, turn.conceptIds)


def createConversationSession(title: str | None = None, model: str | None = None) -> dict:
    conversationId = createConversation(title)
    if model:
        setConversationModel(conversationId, model)
    return {"conversationId": conversationId}


def listConversationSessions() -> dict:
    return {"conversations": listConversations()}


def getConversationSession(conversationId: int) -> dict | None:
    return getConversation(conversationId)


def setConversationSessionModel(conversationId: int, model: str | None) -> dict | None:
    return setConversationModel(conversationId, model)


def loadConversationSession(conversationId: int) -> dict:
    # Touch the cache so the graph is warm for the next turn.
    with lockedGraph(conversationId):
        return buildConversationPayload(conversationId)


def deleteConversationSession(conversationId: int) -> bool:
    deleted = deleteConversation(conversationId)
    invalidate(conversationId)
    forgetConversation(conversationId)
    return deleted


def listConversationTurns(conversationId: int) -> dict:
    return {
        "conversationId": conversationId,
        "turns": serializeStoredTurns(getConversationTurns(conversationId)),
    }


def listConversationGraph(conversationId: int) -> dict:
    storedTurns = getConversationTurns(conversationId)
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
    members = getAllConceptMembers()
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
    members = getAllConceptMembers()
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


def createManualConceptLink(conceptKeyA: str, conceptKeyB: str, kind: str) -> dict | None:
    """Hand-link two concepts. Manual links carry score 1.0 (human-asserted) and
    survive re-analysis of either side as long as both concepts still exist —
    see conceptKey in the persistence model. Returns None if either key is
    unknown or they resolve to the same concept."""
    link = createManualConceptLinkDb(conceptKeyA, conceptKeyB, 1.0, kind)
    return _serializeConceptEdge(link) if link is not None else None


def removeConceptLink(linkId: int) -> bool:
    return deleteConceptLink(linkId)


conversationTitleMaxChars = 60


def _titleFromText(text: str) -> str:
    """First user message, collapsed and trimmed to a word boundary."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= conversationTitleMaxChars:
        return collapsed
    head = collapsed[:conversationTitleMaxChars].rsplit(" ", 1)[0].rstrip()
    return (head or collapsed[:conversationTitleMaxChars].rstrip()) + "…"


def _resolveModelForConversation(conversationId: int, model: str | None) -> str | None:
    """An explicit model on the request wins and becomes the conversation's
    default; otherwise fall back to the conversation's stored choice, then to
    the environment-configured provider (generateAiText / streamAiText's own
    fallback when this returns None)."""
    conversation = getConversation(conversationId)
    if model and (conversation or {}).get("model") != model:
        setConversationModel(conversationId, model)
    return model or (conversation or {}).get("model")


def _finalizeTurn(conversationId: int, userText: str, aiText: str) -> dict:
    """Shared tail of the blocking and streaming send paths: classify and
    persist the turn, auto-title the conversation on its first turn, and
    refresh cross-chat concept links. Assumes the AI reply is already in hand."""
    conversation = getConversation(conversationId)

    with lockedGraph(conversationId) as graph:
        turn = addTurn(graph, userText, aiText)
        persistGraphTurn(conversationId, turn)
        payload = buildConversationPayload(conversationId)

    # This conversation's very first turn: title it from the question, unless
    # it was already given a title (e.g. POST /conversations {"title": ...}).
    if turn.timelineParent is None and not (conversation or {}).get("title"):
        setConversationTitle(conversationId, _titleFromText(userText))

    # Outside the lock: refresh this conversation's cross-chat concept links
    # (debounced, best-effort — never fails the turn).
    maybeRelinkConcepts(conversationId)

    payload["turnId"] = turn.id
    payload["aiText"] = turn.aiText
    return payload


def processChatMessage(conversationId: int, userText: str, model: str | None = None) -> dict:
    modelSpec = _resolveModelForConversation(conversationId, model)
    # Generate the AI reply before taking the lock so the (potentially slow) LLM
    # call does not serialize other work on this conversation's graph.
    aiText = generateAiText(userText, modelSpec)
    return _finalizeTurn(conversationId, userText, aiText)


def streamChatMessage(conversationId: int, userText: str, model: str | None = None):
    """Generator of SSE-ready event dicts for one turn.

    Yields any number of {"type": "delta", "text": "..."} as the reply comes
    in, then exactly one {"type": "done", ...} carrying the same payload shape
    processChatMessage returns (turnId, aiText, turns, nodes, edges). Persists
    nothing until the reply is complete — same ordering as the blocking path,
    just with the network wait broken into visible pieces.
    """
    modelSpec = _resolveModelForConversation(conversationId, model)
    chunks: list[str] = []
    for chunk in streamAiText(userText, modelSpec):
        chunks.append(chunk)
        yield {"type": "delta", "text": chunk}

    aiText = "".join(chunks)
    if not aiText:
        raise RuntimeError("Model returned no text output.")

    payload = _finalizeTurn(conversationId, userText, aiText)
    yield {"type": "done", **payload}


def analyzeConversation(conversationId: int) -> dict:
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


def createConversationEdge(conversationId: int, fromTurnId: int, toTurnId: int, label: str, confidence: float) -> dict:
    edge = createEdge(conversationId, fromTurnId, toTurnId, label, confidence)
    invalidate(conversationId)
    return edge


def patchConversationEdge(edgeId: int, label: str | None = None, confidence: float | None = None) -> dict | None:
    edge = updateEdge(edgeId, label=label, confidence=confidence)
    if edge is not None:
        invalidate(edge["conversationId"])
    return edge


def removeConversationEdge(edgeId: int) -> bool:
    edge = getEdge(edgeId)
    deleted = deleteEdge(edgeId)
    if deleted and edge is not None:
        invalidate(edge["conversationId"])
    return deleted


def getConversationEdge(edgeId: int) -> dict | None:
    return getEdge(edgeId)
