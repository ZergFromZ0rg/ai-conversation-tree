"""Cross-conversation concept linking.

A concept is a set of turns sharing a conceptId within one conversation. This
module scores how close two concepts in *different* conversations are, by
comparing their member turn embeddings, and rewrites the 'auto' rows in
conceptLinks for one conversation at a time.

Concept identity is (conversationId, conceptId). conceptIds are reassigned from
zero on every reanalysis, so these links are disposable: relinkConversation is
called again whenever a conversation changes, and replaceAutoConceptLinks wipes
and rebuilds every 'auto' row touching it.
"""

import logging
import re
import time

import numpy as np

from db import getAllConceptMembers, listAllConceptLinks, replaceAutoConceptLinks

logger = logging.getLogger("conceptIndex")

# Score bands for a concept pair (mean of the top-k pairwise cosines).
sameThreshold = 0.70
relatedThreshold = 0.55

topK = 3
maxLinksPerConcept = 3

# A concept needs at least one turn with this many distinct word tokens to be
# eligible — filters out greeting / acknowledgement concepts.
minSubstantiveTokens = 4

# Longest concept label before it is trimmed on a word boundary.
maxLabelChars = 60

_wordPattern = re.compile(r"[a-z0-9]{2,}")


def _isSubstantive(userText: str) -> bool:
    return len(set(_wordPattern.findall(userText.lower()))) >= minSubstantiveTokens


def _trimLabel(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= maxLabelChars:
        return collapsed
    head = collapsed[:maxLabelChars].rsplit(" ", 1)[0].rstrip()
    return (head or collapsed[:maxLabelChars].rstrip()) + "…"


def conceptLabelsFromMembers(members: list[dict]) -> dict[tuple[int, int], str]:
    """Label each concept by its origin turn's question.

    Picks the concept's earliest root turn (the turn that created the concept),
    falling back to its earliest turn, and trims that user message to
    maxLabelChars on a word boundary.
    """
    best: dict[tuple[int, int], tuple[int, int, str]] = {}
    for row in members:
        key = (row["conversationId"], row["conceptId"])
        # rank 0 = root turn, 1 = inherited; then lowest turnId wins
        candidate = (0 if row["root"] else 1, row["turnId"], row["userText"])
        current = best.get(key)
        if current is None or candidate[:2] < current[:2]:
            best[key] = candidate
    return {key: _trimLabel(value[2]) for key, value in best.items()}


class ConceptProfile:
    __slots__ = ("conversationId", "conceptId", "embeddings", "substantive")

    def __init__(self, conversationId: int, conceptId: int, embeddings: np.ndarray, substantive: bool):
        self.conversationId = conversationId
        self.conceptId = conceptId
        self.embeddings = embeddings  # (m, d) float32, L2-normalised rows
        self.substantive = substantive

    @property
    def key(self) -> tuple[int, int]:
        return (self.conversationId, self.conceptId)


def _normalizeRows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def buildConceptProfiles(members: list[dict]) -> dict[tuple[int, int], ConceptProfile]:
    """Group concept-member rows (from db.getAllConceptMembers) into profiles."""
    vectors: dict[tuple[int, int], list[np.ndarray]] = {}
    substantive: dict[tuple[int, int], bool] = {}

    for row in members:
        key = (row["conversationId"], row["conceptId"])
        vectors.setdefault(key, []).append(
            np.frombuffer(row["embedding"], dtype=np.float32)
        )
        substantive[key] = substantive.get(key, False) or _isSubstantive(row["userText"])

    profiles: dict[tuple[int, int], ConceptProfile] = {}
    for key, rows in vectors.items():
        matrix = _normalizeRows(np.vstack(rows).astype(np.float32))
        profiles[key] = ConceptProfile(key[0], key[1], matrix, substantive[key])
    return profiles


def scoreConceptPair(a: ConceptProfile, b: ConceptProfile) -> float:
    """Mean of the top-k pairwise cosine similarities between member turns.

    Rows are already L2-normalised, so the dot product is the cosine. Top-k
    (not max) keeps one coincidental turn pair from forging a link; not the
    full mean, which would dilute a broad concept.
    """
    sims = (a.embeddings @ b.embeddings.T).ravel()
    if sims.size == 0:
        return 0.0
    k = min(topK, sims.size)
    topValues = np.partition(sims, -k)[-k:]
    return float(topValues.mean())


def labelForScore(score: float) -> str | None:
    if score >= sameThreshold:
        return "same"
    if score >= relatedThreshold:
        return "related"
    return None


def computeLinksForConversation(
    conversationId: int,
    profiles: dict[tuple[int, int], ConceptProfile],
) -> list[tuple[int, int, int, int, float, str]]:
    """Links from one conversation's concepts to every other conversation's.

    Pure: takes the profile map, returns (convA, conceptA, convB, conceptB,
    score, label) tuples, capped at maxLinksPerConcept per source concept.
    """
    targets = [p for p in profiles.values() if p.conversationId == conversationId and p.substantive]
    others = [p for p in profiles.values() if p.conversationId != conversationId and p.substantive]

    links: list[tuple[int, int, int, int, float, str]] = []
    for target in targets:
        scored: list[tuple[float, str, ConceptProfile]] = []
        for other in others:
            score = scoreConceptPair(target, other)
            label = labelForScore(score)
            if label is not None:
                scored.append((score, label, other))
        scored.sort(key=lambda item: item[0], reverse=True)
        for score, label, other in scored[:maxLinksPerConcept]:
            links.append(
                (
                    target.conversationId,
                    target.conceptId,
                    other.conversationId,
                    other.conceptId,
                    score,
                    label,
                )
            )
    return links


def relinkConversation(
    conversationId: int,
    profiles: dict[tuple[int, int], ConceptProfile] | None = None,
) -> int:
    """Recompute and persist the 'auto' concept links touching one conversation.

    Returns the number of links written. Symmetric over time: relinking the
    other conversation later deletes and recreates the same pairs from its side.
    Pass `profiles` to reuse a workspace-wide build across several conversations.
    """
    if profiles is None:
        profiles = buildConceptProfiles(getAllConceptMembers())
    links = computeLinksForConversation(conversationId, profiles)
    replaceAutoConceptLinks(conversationId, links)
    return len(links)


def relinkAllConceptLinks() -> int:
    """Recompute every conversation's 'auto' concept links. Returns the total.

    Builds concept profiles once and reuses them. Used by POST /concepts/relink
    to rebuild after a threshold change or to repair drift.
    """
    profiles = buildConceptProfiles(getAllConceptMembers())
    now = time.monotonic()
    for conversationId in sorted({key[0] for key in profiles}):
        relinkConversation(conversationId, profiles)
        _lastRelink[conversationId] = now
    return len(listAllConceptLinks())


# How long after a relink to skip the next one on the send path. Reanalysis
# forces a relink regardless. Plain dict access under the GIL is good enough
# here: a lost update just means one extra or one skipped relink.
relinkDebounceSeconds = 30.0
_lastRelink: dict[int, float] = {}


def maybeRelinkConcepts(conversationId: int, force: bool = False) -> None:
    """Best-effort relink after a conversation changes.

    Debounced on the send path so a burst of turns relinks once; `force=True`
    (reanalysis) always runs. Never raises — a linking failure must not fail
    the turn or the reanalysis that triggered it.
    """
    now = time.monotonic()
    if not force and now - _lastRelink.get(conversationId, 0.0) < relinkDebounceSeconds:
        return
    try:
        written = relinkConversation(conversationId)
        _lastRelink[conversationId] = now
        logger.debug("relinked conversation %s: %d concept links", conversationId, written)
    except Exception:
        logger.exception("concept relink failed for conversation %s", conversationId)


def forgetConversation(conversationId: int) -> None:
    """Drop debounce state for a deleted conversation (its links cascade in SQL)."""
    _lastRelink.pop(conversationId, None)
