from datetime import datetime
import os
import httpx

from db import (
    createEdge,
    createConversation,
    deleteConversation,
    deleteEdge,
    getConversation,
    getConversationTurns,
    getEdge,
    listConversations,
    listConversationEdges,
    replaceConversationEdges,
    replaceTurnConceptIds,
    saveConceptIds,
    saveSemanticEdges,
    saveTurn,
    saveTurnEmbedding,
    updateTurnMetadata,
    updateEdge,
)
from graphService import addTurn, loadConversationState, reclassifyTurns


def serializeGraphNode(nodeId: int, userText: str, aiText: str, conceptIds: list[int], root: bool, timelineParent: int | None) -> dict:
    return {
        "id": int(nodeId),
        "userText": str(userText),
        "aiText": str(aiText),
        "conceptIds": [int(conceptId) for conceptId in conceptIds],
        "root": bool(root),
        "timelineParent": timelineParent,
    }


def serializeGraphEdge(edgeId: int | None, fromTurnId: int, toTurnId: int, label: str, confidence: float) -> dict:
    return {
        "id": int(edgeId) if edgeId is not None else None,
        "fromTurnId": int(fromTurnId),
        "toTurnId": int(toTurnId),
        "label": str(label),
        "confidence": float(confidence),
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


def buildInMemoryGraphPayload() -> dict:
    from graphService import turns

    nodes = [
        serializeGraphNode(
            nodeId=turn.id,
            userText=turn.userText,
            aiText=turn.aiText,
            conceptIds=turn.conceptIds,
            root=turn.root,
            timelineParent=turn.timelineParent,
        )
        for turn in sorted(turns, key=lambda turn: int(turn.id))
    ]

    edges = []
    syntheticEdgeId = 0
    for turn in sorted(turns, key=lambda turn: int(turn.id)):
        for parentId, label, confidence in sorted(turn.semanticParents, key=lambda parent: (int(parent[0]), str(parent[1]), float(parent[2]))):
            edges.append(
                serializeGraphEdge(
                    edgeId=syntheticEdgeId,
                    fromTurnId=parentId,
                    toTurnId=turn.id,
                    label=label,
                    confidence=confidence,
                )
            )
            syntheticEdgeId += 1

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
    message = responseBody.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    return ""


def generateAiText(userText: str) -> str:
    ollamaModelName = os.environ.get("OLLAMA_MODEL")
    if ollamaModelName:
        ollamaBaseUrl = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        response = httpx.post(
            f"{ollamaBaseUrl}/api/chat",
            json={
                "model": ollamaModelName,
                "messages": [
                    {
                        "role": "user",
                        "content": userText,
                    }
                ],
                "stream": False,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        responseBody = response.json()
        aiText = extractOllamaText(responseBody)
        if not aiText:
            raise RuntimeError("Ollama returned no text output.")
        return aiText

    if os.environ.get("AI_CONVERSATION_TREE_STUB_LLM", "1") == "1":
        return f"Stub LLM response for: {userText}"

    apiKey = os.environ.get("OPENAI_API_KEY")
    if not apiKey:
        raise RuntimeError("Set OLLAMA_MODEL, set OPENAI_API_KEY, or enable AI_CONVERSATION_TREE_STUB_LLM=1.")

    modelName = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
    response = httpx.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {apiKey}",
            "Content-Type": "application/json",
        },
        json={
            "model": modelName,
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
    responseBody = response.json()
    aiText = extractResponseText(responseBody)
    if not aiText:
        raise RuntimeError("LLM returned no text output.")
    return aiText


def buildConversationPayload(conversationId: int) -> dict:
    storedTurns = getConversationTurns(conversationId)
    responseTurns = [
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
    graph = buildGraphPayloadFromStoredTurns(conversationId, storedTurns)
    return {
        "conversationId": conversationId,
        "turns": responseTurns,
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
    saveTurnEmbedding(conversationId, turn.id, turn.embedding.tolist())
    saveSemanticEdges(conversationId, turn.id, turn.semanticParents)
    saveConceptIds(conversationId, turn.id, turn.conceptIds)


def createConversationSession(title: str | None = None) -> dict:
    conversationId = createConversation(title)
    return {"conversationId": conversationId}


def listConversationSessions() -> dict:
    return {"conversations": listConversations()}


def getConversationSession(conversationId: int) -> dict | None:
    return getConversation(conversationId)


def loadConversationSession(conversationId: int) -> dict:
    loadConversationState(conversationId)
    return buildConversationPayload(conversationId)


def deleteConversationSession(conversationId: int) -> bool:
    return deleteConversation(conversationId)


def listConversationTurns(conversationId: int) -> dict:
    return {"conversationId": conversationId, "turns": buildConversationPayload(conversationId)["turns"]}


def listConversationGraph(conversationId: int) -> dict:
    storedTurns = getConversationTurns(conversationId)
    graph = buildGraphPayloadFromStoredTurns(conversationId, storedTurns)
    return {"conversationId": conversationId, "nodes": graph["nodes"], "edges": graph["edges"]}


def processChatMessage(conversationId: int, userText: str) -> dict:
    loadConversationState(conversationId)
    aiText = generateAiText(userText)
    turn = addTurn(userText, aiText)

    persistGraphTurn(conversationId, turn)

    payload = buildConversationPayload(conversationId)
    payload["turnId"] = turn.id
    payload["aiText"] = turn.aiText
    return payload


def analyzeConversation(conversationId: int) -> dict:
    loadConversationState(conversationId)
    rebuiltTurns = reclassifyTurns()

    replaceConversationEdges(
        conversationId,
        [
            (parentId, turn.id, label, confidence)
            for turn in rebuiltTurns
            for parentId, label, confidence in turn.semanticParents
        ],
    )

    for turn in rebuiltTurns:
        updateTurnMetadata(
            conversationId=conversationId,
            turnId=turn.id,
            root=turn.root,
            timelineParent=turn.timelineParent,
        )
        replaceTurnConceptIds(conversationId, turn.id, turn.conceptIds)

    payload = buildConversationPayload(conversationId)
    return {
        "conversationId": conversationId,
        "turnCount": len(payload["turns"]),
        "edgeCount": len(payload["edges"]),
        "graph": {"nodes": payload["nodes"], "edges": payload["edges"]},
    }


def createConversationEdge(conversationId: int, fromTurnId: int, toTurnId: int, label: str, confidence: float) -> dict:
    return createEdge(conversationId, fromTurnId, toTurnId, label, confidence)


def patchConversationEdge(edgeId: int, label: str | None = None, confidence: float | None = None) -> dict | None:
    return updateEdge(edgeId, label=label, confidence=confidence)


def removeConversationEdge(edgeId: int) -> bool:
    return deleteEdge(edgeId)


def getConversationEdge(edgeId: int) -> dict | None:
    return getEdge(edgeId)
