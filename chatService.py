from datetime import datetime
import os
import httpx

from db import (
    applyReclassification,
    createEdge,
    createConversation,
    deleteConversation,
    deleteEdge,
    getConversation,
    getConversationTurns,
    getEdge,
    listConversations,
    listConversationEdges,
    saveConceptIds,
    saveSemanticEdges,
    saveTurn,
    updateEdge,
)
from graphService import addTurn, reclassifyTurns
from graphStore import invalidate, lockedGraph
import vectorStore


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
    vectorStore.saveEmbedding(conversationId, turn.id, turn.embedding.tolist())
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
    # Touch the cache so the graph is warm for the next turn.
    with lockedGraph(conversationId):
        return buildConversationPayload(conversationId)


def deleteConversationSession(conversationId: int) -> bool:
    deleted = deleteConversation(conversationId)
    vectorStore.deleteConversation(conversationId)
    invalidate(conversationId)
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


def processChatMessage(conversationId: int, userText: str) -> dict:
    # Generate the AI reply before taking the lock so the (potentially slow) LLM
    # call does not serialize other work on this conversation's graph.
    aiText = generateAiText(userText)

    with lockedGraph(conversationId) as graph:
        turn = addTurn(graph, userText, aiText)
        persistGraphTurn(conversationId, turn)
        payload = buildConversationPayload(conversationId)

    payload["turnId"] = turn.id
    payload["aiText"] = turn.aiText
    return payload


def analyzeConversation(conversationId: int) -> dict:
    with lockedGraph(conversationId) as graph:
        rebuiltTurns = reclassifyTurns(graph)

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
        )

        payload = buildConversationPayload(conversationId)
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
