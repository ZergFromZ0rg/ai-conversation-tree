"""Context-loss detection.

Long chats fail quietly: the model stops seeing something said fifty turns
ago, and the user finds out only when the answers go wrong. This module reads
the conversation's own structure for the tells:

- ``reask``      the user repeats an earlier question — the earlier answer
                 didn't stick, or the user could no longer find it
- ``memoryCue``  the user points back ("as I said", "you forgot", "remember
                 that…") — an explicit complaint about lost context
- ``farReach``   a turn depends (via a semantic edge) on a turn far behind it;
                 when the text in between exceeds the context window the model
                 almost certainly can no longer see the thing it builds on

Everything here is deterministic. ``analyzeContextHealth`` is pure (it takes
the vectors as an argument) so it can be tested without loading any model.
"""

import re

import numpy as np

from .config import DEFAULT_CONTEXT_WINDOW_TOKENS
from .conversationErrors import NotFoundError
from .providers import contextWindowFor

# Tokens are estimated as chars / 4. How large a window to assume comes from
# Settings.contextWindowTokens (a larger or smaller model needs a different one).
charsPerToken = 4
reaskSimilarity = 0.80
immediateReaskSimilarity = 0.90
farReachTurns = 4

_memoryCuePattern = re.compile(
    r"\b("
    r"as i (said|mentioned|told|noted)|like i (said|mentioned|told)|"
    r"i (already|just) (said|told|mentioned|asked|explained)|"
    r"you (forgot|ignored|missed|keep (forgetting|ignoring))|"
    r"(we|you) (already|just) (covered|discussed|did|said|answered)|"
    r"(as|like) (we|you) (discussed|said|covered|agreed)|"
    r"(i|we) (said|mentioned|discussed|asked) (earlier|before|above|previously)|"
    r"earlier (i|you|we)|remember (that|when|what|how)|"
    r"go back to|back to (the|my|what|our)|"
    r"(the|that) (thing|question|example|code) (i|we) (had|mentioned|asked)"
    r")\b",
    re.IGNORECASE,
)

_signalWeights = {"reask": 0.5, "memoryCue": 0.45, "farReach": 0.6, "longRange": 0.25}


def estimateTokens(text: str) -> int:
    return max(1, len(text) // charsPerToken)


def recentHistory(turns: list[dict], newUserText: str, windowTokens: int) -> list[dict]:
    """The earlier turns to send a model along with a new message.

    Returned as chat messages (``{"role": "user" | "assistant", "content": str}``),
    oldest first. Whole turns only, taken from the newest backwards until the
    window is full, so what a model is sent matches the horizon that
    ``analyzeContextHealth`` draws (same token estimate). The new message is
    charged against the window first. Stops at the first turn that doesn't fit
    rather than skipping it, so the history is always contiguous.

    `turns` are dicts with `userText` and `aiText` in timeline order.
    """
    budget = windowTokens - estimateTokens(newUserText)
    kept: list[dict] = []
    used = 0
    for turn in reversed(turns):
        cost = estimateTokens(turn["userText"]) + estimateTokens(turn["aiText"])
        if used + cost > budget:
            break
        used += cost
        kept.append(turn)
    kept.reverse()
    return [
        message
        for turn in kept
        for message in (
            {"role": "user", "content": turn["userText"]},
            {"role": "assistant", "content": turn["aiText"]},
        )
    ]


def _combine(weights: list[float]) -> float:
    remaining = 1.0
    for weight in weights:
        remaining *= 1.0 - weight
    return 1.0 - remaining


def _level(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def analyzeContextHealth(
    turns: list[dict],
    userVectors: np.ndarray,
    edges: list[dict],
    windowTokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
) -> dict:
    """Score each turn for signs that earlier context was lost.

    `turns` are dicts with `id`, `userText`, `aiText` in timeline order;
    `userVectors[i]` is the (L2-normalised) embedding of turns[i]'s user text;
    `edges` are dicts with `fromTurnId`, `toTurnId`, `label`.
    """
    tokens = [estimateTokens(turn["userText"]) + estimateTokens(turn["aiText"]) for turn in turns]
    prefix = [0]
    for count in tokens:
        prefix.append(prefix[-1] + count)
    positionById = {turn["id"]: index for index, turn in enumerate(turns)}

    similarity = userVectors @ userVectors.T if len(turns) else np.zeros((0, 0))
    flagged = []
    for index, turn in enumerate(turns):
        signals: list[dict] = []

        for earlier in range(index):
            gap = index - earlier
            threshold = immediateReaskSimilarity if gap == 1 else reaskSimilarity
            if similarity[index, earlier] >= threshold:
                signals.append(
                    {
                        "kind": "reask",
                        "weight": _signalWeights["reask"],
                        "turnId": turns[earlier]["id"],
                        "detail": f"Repeats the question from turn {turns[earlier]['id']}",
                    }
                )
                break

        if index > 0 and _memoryCuePattern.search(turn["userText"]):
            signals.append(
                {
                    "kind": "memoryCue",
                    "weight": _signalWeights["memoryCue"],
                    "turnId": None,
                    "detail": "Points back to something said earlier",
                }
            )

        # Farthest back-reference through a semantic edge into this turn.
        reach = None
        for edge in edges:
            if edge["toTurnId"] != turn["id"] or edge["fromTurnId"] not in positionById:
                continue
            fromIndex = positionById[edge["fromTurnId"]]
            if index - fromIndex >= farReachTurns and (reach is None or fromIndex < reach):
                reach = fromIndex
        if reach is not None:
            between = prefix[index] - prefix[reach + 1] + estimateTokens(turns[reach]["aiText"])
            if between >= windowTokens:
                signals.append(
                    {
                        "kind": "farReach",
                        "weight": _signalWeights["farReach"],
                        "turnId": turns[reach]["id"],
                        "detail": (
                            f"Builds on turn {turns[reach]['id']}, about {between:,} tokens back — "
                            "likely outside the model's window"
                        ),
                    }
                )
            else:
                signals.append(
                    {
                        "kind": "longRange",
                        "weight": _signalWeights["longRange"],
                        "turnId": turns[reach]["id"],
                        "detail": f"Builds on turn {turns[reach]['id']}, {index - reach} turns back",
                    }
                )

        if signals:
            score = _combine([signal["weight"] for signal in signals])
            flagged.append(
                {
                    "turnId": turn["id"],
                    "score": round(score, 3),
                    "level": _level(score),
                    "signals": [
                        {key: value for key, value in signal.items() if key != "weight"} for signal in signals
                    ],
                }
            )

    # The horizon: the earliest turn that still fits in the window when
    # counting back from the newest one. Turns before it are outside what a
    # model fed this whole chat would weigh.
    horizonTurnId = None
    for index in range(len(turns)):
        if prefix[-1] - prefix[index] <= windowTokens:
            if index > 0:
                horizonTurnId = turns[index]["id"]
            break

    levels = [item["level"] for item in flagged]
    if "high" in levels or levels.count("medium") >= 2:
        overall = "high"
    elif "medium" in levels or len(levels) >= 2:
        overall = "medium"
    else:
        overall = "low"

    return {
        "overall": overall,
        "turnCount": len(turns),
        "estimatedTokens": prefix[-1],
        "windowTokens": windowTokens,
        "windowPressure": round(prefix[-1] / windowTokens, 2) if windowTokens else 0.0,
        "horizonTurnId": horizonTurnId,
        "flaggedTurns": flagged,
    }


def buildContextHealth(conversationId: int, windowTokens: int | None = None) -> dict:
    from .db import getConversation, getConversationTurns, listConversationEdges
    from .graphService import encodeText

    conversation = getConversation(conversationId)
    if conversation is None:
        raise NotFoundError("Conversation not found.")
    turns = getConversationTurns(conversationId, withEmbeddings=False)
    edges = listConversationEdges(conversationId)
    if turns:
        vectors = np.stack([np.asarray(encodeText(turn["userText"]), dtype=np.float32) for turn in turns])
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    else:
        vectors = np.zeros((0, 1), dtype=np.float32)
    report = analyzeContextHealth(turns, vectors, edges, windowTokens or contextWindowFor(conversation.get("model")))
    report["conversationId"] = conversationId
    return report
