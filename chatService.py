from datetime import datetime
import os
import httpx

from db import (
    createConversation,
    getConversationTurns,
    listConversations,
    saveConceptIds,
    saveSemanticEdges,
    saveTurn,
    saveTurnEmbedding,
)
from graphService import addTurn, loadConversationState, turns


def extractResponseText(responseBody: dict) -> str:
    outputItems = responseBody.get("output", [])
    textParts: list[str] = []

    for outputItem in outputItems:
        for contentItem in outputItem.get("content", []):
            if contentItem.get("type") == "output_text":
                textParts.append(contentItem.get("text", ""))

    return "\n".join(part for part in textParts if part).strip()


def generateAiText(userText: str) -> str:
    if os.environ.get("AI_CONVERSATION_TREE_STUB_LLM", "1") == "1":
        return f"Stub LLM response for: {userText}"

    apiKey = os.environ.get("OPENAI_API_KEY")
    if not apiKey:
        raise RuntimeError("Set OPENAI_API_KEY or enable AI_CONVERSATION_TREE_STUB_LLM=1.")

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
    nodes = [
        {
            "id": turn["id"],
            "userText": turn["userText"],
            "aiText": turn["aiText"],
            "conceptIds": turn["conceptIds"],
            "root": turn["root"],
            "timelineParent": turn["timelineParent"],
        }
        for turn in storedTurns
    ]
    edges = [
        {
            "from": parentId,
            "to": turn["id"],
            "type": label,
            "confidence": confidence,
        }
        for turn in storedTurns
        for parentId, label, confidence in turn["semanticParents"]
    ]
    return {
        "conversationId": conversationId,
        "turns": responseTurns,
        "nodes": nodes,
        "edges": edges,
    }


def createConversationSession(title: str | None = None) -> dict:
    conversationId = createConversation(title)
    return {"conversationId": conversationId}


def listConversationSessions() -> dict:
    return {"conversations": listConversations()}


def loadConversationSession(conversationId: int) -> dict:
    loadConversationState(conversationId)
    return buildConversationPayload(conversationId)


def processChatMessage(conversationId: int, userText: str) -> dict:
    loadConversationState(conversationId)
    aiText = generateAiText(userText)
    turn = addTurn(userText, aiText)

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

    payload = buildConversationPayload(conversationId)
    payload["turnId"] = turn.id
    payload["aiText"] = turn.aiText
    return payload
