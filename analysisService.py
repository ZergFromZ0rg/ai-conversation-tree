"""Conversation summaries, workspace search, and exportable reports."""

from collections import Counter
from datetime import datetime
import re

import numpy as np

from db import (
    getConversation,
    getConversationTurns,
    getAllSearchTurns,
    getAllTurnEmbeddings,
    listConversationEdges,
)
from graphService import encodeText
from vectorStore import searchWorkspace

_wordPattern = re.compile(r"[a-z0-9][a-z0-9'_-]{1,}", re.IGNORECASE)
_stopwords = frozenset(
    """a an the and or but if then than so of to in on at by for with about from into through
    what which who how why is are was were be been being have has had do does did can could
    would should i me my we our you your this that these those it its they their tell explain
    please want need get make use used using"""
    .split()
)
minimumSearchScore = 0.20


def _terms(text: str) -> list[str]:
    return [word.lower() for word in _wordPattern.findall(text) if word.lower() not in _stopwords]


def _shorten(text: str, limit: int = 180) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"


def _topicTerms(turns: list[dict], limit: int = 8) -> list[str]:
    counts = Counter(term for turn in turns for term in _terms(turn["userText"]))
    return [term for term, _ in counts.most_common(limit)]


def buildConversationSummary(conversationId: int) -> dict:
    conversation = getConversation(conversationId)
    if conversation is None:
        return None
    turns = getConversationTurns(conversationId)
    edges = listConversationEdges(conversationId)
    roots = [turn for turn in turns if turn["root"]]
    topics = _topicTerms(turns)
    topicText = ", ".join(topics) if topics else "general discussion"
    if not turns:
        summary = "This conversation has no turns yet."
    else:
        opening = _shorten(turns[0]["userText"])
        summary = (
            f"This conversation contains {len(turns)} turns across {len(roots)} topic thread"
            f"{'s' if len(roots) != 1 else ''}. It began with “{opening}” and mainly discusses "
            f"{topicText}. The graph contains {len(edges)} semantic relationship"
            f"{'s' if len(edges) != 1 else ''}."
        )
    return {
        "conversationId": conversationId,
        "title": conversation["title"],
        "summary": summary,
        "turnCount": len(turns),
        "threadCount": len(roots),
        "edgeCount": len(edges),
        "topics": topics,
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "mode": "structured-local",
    }


def searchConversations(query: str, limit: int = 20) -> dict:
    query = query.strip()
    if not query:
        return {"query": query, "results": []}
    embedding = encodeText(query)
    vectorRows = searchWorkspace(embedding, limit)
    if vectorRows is None:
        queryVector = np.asarray(embedding, dtype=np.float32)
        queryNorm = max(float(np.linalg.norm(queryVector)), 1e-8)
        scored = []
        for row in getAllTurnEmbeddings():
            candidate = np.frombuffer(row["embedding"], dtype=np.float32)
            score = float(candidate @ queryVector) / max(float(np.linalg.norm(candidate)) * queryNorm, 1e-8)
            scored.append((row["conversationId"], row["turnId"], score))
        vectorRows = sorted(scored, key=lambda item: item[2], reverse=True)[:limit]
    turnsByKey = {
        (row["conversationId"], row["turnId"]): row for row in getAllSearchTurns()
    }
    results = []
    for conversationId, turnId, score in vectorRows:
        if score < minimumSearchScore:
            continue
        row = turnsByKey.get((conversationId, turnId))
        if row is None:
            continue
        results.append(
            {
                "conversationId": conversationId,
                "conversationTitle": row["conversationTitle"],
                "turnId": turnId,
                "userText": row["userText"],
                "aiText": row["aiText"],
                "score": round(float(score), 4),
            }
        )
    return {"query": query, "results": results}


def buildConversationReport(conversationId: int) -> dict | None:
    conversation = getConversation(conversationId)
    if conversation is None:
        return None
    turns = getConversationTurns(conversationId)
    edges = listConversationEdges(conversationId)
    summary = buildConversationSummary(conversationId)
    return {
        "conversation": conversation,
        "summary": summary,
        "turns": [
            {
                "turnId": turn["id"],
                "userText": turn["userText"],
                "aiText": turn["aiText"],
                "root": turn["root"],
                "timelineParent": turn["timelineParent"],
                "conceptIds": turn["conceptIds"],
            }
            for turn in turns
        ],
        "edges": edges,
        "topics": _topicTerms(turns),
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
    }


def reportAsMarkdown(report: dict) -> str:
    conversation = report["conversation"]
    summary = report["summary"]
    title = conversation["title"] or f"Conversation {conversation['id']}"
    lines = [
        f"# {title}",
        "",
        f"Generated: {report['generatedAt']}",
        "",
        "## Summary",
        "",
        summary["summary"],
        "",
        "## Topics",
        "",
        ", ".join(report["topics"]) or "No recurring topic terms detected.",
        "",
        "## Conversation turns",
        "",
    ]
    for turn in report["turns"]:
        marker = " — root thread" if turn["root"] else ""
        lines.extend(
            [
                f"### Turn {turn['turnId']}{marker}",
                "",
                f"**User:** {turn['userText']}",
                "",
                f"**Assistant:** {turn['aiText']}",
                "",
            ]
        )
    lines.extend(["## Semantic edges", ""])
    if report["edges"]:
        lines.extend(
            f"- Turn {edge['fromTurnId']} → Turn {edge['toTurnId']}: {edge['label']} "
            f"(confidence {edge['confidence']:.2f}, {edge['origin']})"
            for edge in report["edges"]
        )
    else:
        lines.append("No semantic edges recorded.")
    return "\n".join(lines) + "\n"
