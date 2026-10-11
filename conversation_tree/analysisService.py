"""Conversation summaries, workspace search, and exportable reports."""

from collections import Counter
from datetime import datetime
import hashlib
import re
import threading

import numpy as np

from .conversationErrors import InvalidInputError, NotFoundError
from .db import (
    getConversation,
    getConversationTurns,
    getAllSearchTurns,
    getAllTurnEmbeddings,
    listConversationEdges,
)
from .graphService import encodeText
from .providers import STUB_MODEL_SPEC, ReplyRequest, getReplier, requireValidModelSpec
from .vectorStore import searchConversation, searchWorkspace

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


def _loadConversation(conversationId: int) -> tuple[dict, list[dict], list[dict]]:
    """A conversation, its turns (text only), and its edges, each read once."""
    conversation = getConversation(conversationId)
    if conversation is None:
        raise NotFoundError("Conversation not found.")
    return (
        conversation,
        getConversationTurns(conversationId, withEmbeddings=False),
        listConversationEdges(conversationId),
    )


def _summaryFrom(conversation: dict, turns: list[dict], edges: list[dict]) -> dict:
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
        "conversationId": conversation["id"],
        "title": conversation["title"],
        "summary": summary,
        "turnCount": len(turns),
        "threadCount": len(roots),
        "edgeCount": len(edges),
        "topics": topics,
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "mode": "structured-local",
    }


def buildConversationSummary(conversationId: int) -> dict:
    return _summaryFrom(*_loadConversation(conversationId))


def _matchedTerms(queryTerms: list[str], text: str) -> list[str]:
    lowered = text.lower()
    return [term for term in queryTerms if term in lowered]


def _snippet(text: str, terms: list[str], radius: int = 110) -> str:
    """A window of `text` around its first matched term (or its opening)."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= radius * 2:
        return collapsed
    lowered = collapsed.lower()
    anchor = min((at for at in (lowered.find(term) for term in terms) if at >= 0), default=0)
    start = max(anchor - radius // 2, 0)
    end = min(start + radius * 2, len(collapsed))
    return ("…" if start > 0 else "") + collapsed[start:end].strip() + ("…" if end < len(collapsed) else "")


MAX_SEARCH_LIMIT = 100


def searchConversations(query: str, limit: int = 20, conversationId: int | None = None) -> dict:
    """Hybrid search across every stored turn.

    Semantic similarity (embedding cosine) finds turns that talk about the same
    thing in different words; a lexical pass rewards turns that contain the
    query's own words and tells the UI which words to highlight. Each hit says
    whether the match was in the question or the answer, carries a snippet
    around the match, and its position in the conversation so the viewer can
    jump straight to the turn.

    Raises InvalidInputError for a blank query or a limit outside 1..MAX_SEARCH_LIMIT.
    """
    query = query.strip()
    if not query:
        raise InvalidInputError("query must not be empty.")
    if not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise InvalidInputError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}.")
    queryTerms = list(dict.fromkeys(_terms(query)))
    embedding = encodeText(query)
    if conversationId is None:
        vectorRows = searchWorkspace(embedding, limit * 4)
    else:
        inConversation = searchConversation(conversationId, embedding, None, 100000)
        vectorRows = (
            None
            if inConversation is None
            else [(conversationId, turnId, score) for turnId, score in inConversation]
        )
    if vectorRows is None:
        queryVector = np.asarray(embedding, dtype=np.float32)
        queryNorm = max(float(np.linalg.norm(queryVector)), 1e-8)
        scored = []
        for row in getAllTurnEmbeddings(conversationId):
            candidate = np.frombuffer(row["embedding"], dtype=np.float32)
            score = float(candidate @ queryVector) / max(float(np.linalg.norm(candidate)) * queryNorm, 1e-8)
            scored.append((row["conversationId"], row["turnId"], score))
        vectorRows = scored
    semanticByKey = {(cid, tid): score for cid, tid, score in vectorRows}

    allTurns = getAllSearchTurns(conversationId)
    turnCounts = Counter(row["conversationId"] for row in allTurns)
    candidates = []
    for row in allTurns:
        key = (row["conversationId"], row["turnId"])
        semantic = semanticByKey.get(key, 0.0)
        questionTerms = _matchedTerms(queryTerms, row["userText"])
        answerTerms = _matchedTerms(queryTerms, row["aiText"])
        matched = list(dict.fromkeys(questionTerms + answerTerms))
        lexical = len(matched) / len(queryTerms) if queryTerms else 0.0
        if semantic < minimumSearchScore and not matched:
            continue
        # Weight a question hit over an answer hit: the user's own phrasing is
        # the better "where did I ask about this" signal.
        questionBoost = 0.04 if questionTerms else 0.0
        final = 0.8 * max(semantic, 0.0) + 0.2 * lexical + questionBoost
        inQuestion = len(questionTerms) >= len(answerTerms)
        text = row["userText"] if inQuestion else row["aiText"]
        candidates.append(
            {
                "conversationId": row["conversationId"],
                "conversationTitle": row["conversationTitle"],
                "source": row["source"],
                "turnId": row["turnId"],
                "turnCount": turnCounts[row["conversationId"]],
                "userText": row["userText"],
                "aiText": row["aiText"],
                "matchIn": "question" if inQuestion else "answer",
                "snippet": _snippet(text, matched),
                "matchedTerms": matched,
                "semanticScore": round(float(semantic), 4),
                "score": round(min(float(final), 1.0), 4),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return {"query": query, "terms": queryTerms, "results": candidates[:limit]}


_summaryCache: dict[tuple, dict] = {}
_summaryCacheLock = threading.Lock()
_SUMMARY_CACHE_MAX = 64
_summaryPrompt = (
    "Summarize the conversation below in 3-5 sentences of plain prose. Say what the user was "
    "trying to do, the main conclusions or answers reached, and any threads left open. "
    "No preamble, no bullet points, no markdown headings.\n\n"
)
_summaryTranscriptChars = 12000


def _summaryCacheKey(conversationId: int, turns: list[dict], modelSpec: str | None, replier) -> tuple:
    """Identifies a summary by everything it was written from: the whole
    transcript (not just its length and last turn, so an edited earlier turn is
    a different summary), the model, and the replier that wrote it."""
    digest = hashlib.blake2b(digest_size=16)
    for turn in turns:
        digest.update(turn["userText"].encode())
        digest.update(b"\x00")
        digest.update(turn["aiText"].encode())
        digest.update(b"\x01")
    return (conversationId, digest.hexdigest(), modelSpec, id(replier))


def _cachedSummary(key: tuple) -> dict | None:
    with _summaryCacheLock:
        return _summaryCache.get(key)


def _cacheSummary(key: tuple, summary: dict) -> None:
    with _summaryCacheLock:
        _summaryCache[key] = summary
        while len(_summaryCache) > _SUMMARY_CACHE_MAX:
            _summaryCache.pop(next(iter(_summaryCache)))


def _transcriptForSummary(turns: list[dict]) -> str:
    lines = []
    for turn in turns:
        lines.append(f"User: {_shorten(turn['userText'], 600)}")
        lines.append(f"Assistant: {_shorten(turn['aiText'], 600)}")
    transcript = "\n".join(lines)
    if len(transcript) > _summaryTranscriptChars:
        # Keep the opening (the goal) and the end (where it landed).
        half = _summaryTranscriptChars // 2
        transcript = transcript[:half] + "\n[...middle of the conversation omitted...]\n" + transcript[-half:]
    return transcript


def buildLlmSummary(conversationId: int, model: str | None = None, apiKey: str | None = None) -> dict:
    """The structured summary with its prose replaced by a model-written one.

    Falls back to the local summary (mode "structured-local", with the reason)
    when the stub provider is selected or the provider fails, so the caller
    always gets a usable summary. Raises InvalidInputError for a malformed
    model spec and NotFoundError for an unknown conversation.
    """
    requireValidModelSpec(model)
    replier = getReplier()
    conversation, turns, edges = _loadConversation(conversationId)
    base = _summaryFrom(conversation, turns, edges)
    if not turns:
        return base
    modelSpec = model or conversation.get("model") or replier.defaultModelSpec()
    if modelSpec == STUB_MODEL_SPEC:
        return {**base, "fallbackReason": "The stub model can't write summaries; pick a real model."}

    cacheKey = _summaryCacheKey(conversationId, turns, modelSpec, replier)
    cached = _cachedSummary(cacheKey)
    if cached is not None:
        return cached

    try:
        # No conversationId and no history: the prompt already contains the transcript.
        text = replier.generate(
            ReplyRequest(_summaryPrompt + _transcriptForSummary(turns), modelSpec, apiKey)
        ).strip()
    except Exception as error:  # provider errors vary widely; never fail the panel
        return {**base, "fallbackReason": f"Model call failed: {error}"}
    if not text:
        return {**base, "fallbackReason": "The model returned no text."}
    result = {**base, "summary": text, "mode": "llm", "model": modelSpec}
    _cacheSummary(cacheKey, result)
    return result


def reset() -> None:
    """Drop cached model-written summaries (keyed by conversation id, which means
    nothing in a different database); called by init() when the settings change."""
    with _summaryCacheLock:
        _summaryCache.clear()


def buildConversationReport(conversationId: int) -> dict:
    conversation, turns, edges = _loadConversation(conversationId)
    return {
        "conversation": conversation,
        "summary": _summaryFrom(conversation, turns, edges),
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
