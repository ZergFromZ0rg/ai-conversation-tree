from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from db import getConversationTurns
from models import TurnModel

import numpy as np
import re
from sentence_transformers import CrossEncoder, SentenceTransformer

biEncoderModelName = 'all-MiniLM-L6-v2'
crossEncoderModelName = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

model = SentenceTransformer(biEncoderModelName)
crossEncoder = CrossEncoder(crossEncoderModelName)


@dataclass
class ConversationGraph:
    """One conversation's in-memory turn graph.

    Every function that reads or mutates this state takes the graph explicitly;
    there is no module-global graph. Callers that mutate a graph must serialize
    on the per-conversation lock in graphStore.
    """

    turns: list[TurnModel] = field(default_factory=list)
    conceptCounter: int = 0

THRESHOLD = 0.55
continuationThreshold = 0.55
closeGap = 0.05
weakThreshold = 0.35
minEmbeddingFloor = 0.12
branchMinScore = 1.0
continuationMinScore = 1.2
relatedMinScore = 2.25
relatedSimilarityThreshold = THRESHOLD
# The final "is this confidence high enough to commit to a label" bar
# (selectEdgeLabel and both immediate/older-turn candidate checks). Was 0.5;
# a real pronoun-reference follow-up ("what makes that book stand out?")
# computed a genuine, reasoned branch confidence of 0.491 — just under the
# old bar — once the _maxTopicalSignal-based gate loosening let it be scored
# at all. Lowered with real margin below that value (not shaved to the
# minimum that passes it) so this isn't fit to one example.
edgeConfidenceThreshold = 0.45
candidateSignalThreshold = 0.52
crossCandidateThreshold = 0.5
parallelDefinitionRelatedThreshold = 0.3
maxParents = 2
shortMsgWords = 5
maxOlderCandidates = 5
olderTurnDecay = 0.005
strongTopicSimilarityThreshold = 0.25
strongContentOverlapThreshold = 0.18
subthreadContinuationSimilarityThreshold = 0.48
subthreadTopicSimilarityThreshold = 0.5
subthreadLexicalOverlapThreshold = 0.15

# Discourse markers for classification (improved patterns)
clarificationPatterns = [
    "what do you mean",
    "can you clarify",
    "explain",
    "how so",
    "why",
    "does that mean",
]

referencePatterns = [
    "you said",
    "you mentioned",
    "above",
    "previous",
]

pronounReferencePatterns = [
    "that",
    "this",
    "it",
]

forwardPatterns = [
    "how do i",
    "how would i",
    "how do we",
    "implement",
    "build",
    "use",
    "learn",
    "learning",
    "go about",
    "next",
    "what's next",
    "should i",
    "best way",
    "tell me more",
    "more about",
    "go deeper",
    "continue",
]

topicShiftPatterns = [
    "what about",
    "similar to",
    "like",
    "alternative",
    "instead",
    "another option",
]

comparisonPatterns = [
    "difference between",
    "differences between",
    "compare",
    "compared to",
    "compared with",
    "vs",
    "versus",
]

followupPatterns = [
    "what if",
    "can i",
    "should i",
    "when do i",
    "when should i",
    "where do i",
    "which one",
]

# Explicit "next step in the same task" markers. Paired with a forward pattern
# ("how do i ... next"), these anchor a continuation even when the embeddings
# have drifted, because the phrasing itself asserts sequence.
sequencingPatterns = [
    "next",
    "next step",
    "after that",
    "afterwards",
    "from here",
    "what's next",
    "whats next",
]

definitionPrefixes = (
    "what is ",
    "what are ",
)

stopwords = {
    "a",
    "an",
    "and",
    "are",
    "between",
    "do",
    "does",
    "for",
    "how",
    "i",
    "is",
    "of",
    "the",
    "to",
    "what",
    "whats",
}

# Basic text helpers

def nextId(graph: ConversationGraph) -> int:
    return len(graph.turns)


@lru_cache(maxsize=2048)
def encodeText(text: str):
    return model.encode(text)


def normalizeWord(word: str) -> str:
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def contentTerms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {normalizeWord(word) for word in words if word not in stopwords}


def countPatternMatches(text: str, patterns: list[str]) -> int:
    # Match whole phrases so short tokens like "it" do not fire inside unrelated words.
    count = 0
    for pattern in patterns:
        escaped = re.escape(pattern).replace(r"\ ", r"\s+")
        if re.search(rf"\b{escaped}\b", text):
            count += 1
    return count


# Discourse signals and gating

def hasComparisonWithPriorTopic(features: dict) -> bool:
    return features["comparisonScore"] > 0 and features["contentOverlap"] > 0


def hasComparisonFollowup(features: dict) -> bool:
    # A comparison only counts when it is anchored to the prior topic.
    return (
        features["comparisonScore"] > 0
        and (
            features["contentOverlap"] > 0
            or features["pronounReferenceScore"] > 0
            or features["userTopicSimilarity"] >= parallelDefinitionRelatedThreshold
        )
    )


def _maxTopicalSignal(similarity: float, features: dict) -> float:
    return max(
        similarity,
        features["userTopicSimilarity"],
        features["answerSimilarity"],
        features["contentOverlap"],
    )


def hasForwardAnchor(similarity: float, features: dict) -> bool:
    # Generic "how do I learn/use X" phrasing should only count as continuation
    # when topic identity is clearly established, not from incidental overlap.
    # An explicit sequencing marker ("...next", "after that") lowers that bar
    # from "clearly established" to "not absent" — the phrasing asserts a next
    # step, but the topic still has to be at least faintly present so a plain
    # topic switch that happens to contain "next" is not pulled in.
    if features["sequencingScore"] > 0 and _maxTopicalSignal(similarity, features) >= minEmbeddingFloor:
        return True
    return (
        similarity >= strongTopicSimilarityThreshold
        or features["userTopicSimilarity"] >= strongTopicSimilarityThreshold
        or features["answerSimilarity"] >= strongTopicSimilarityThreshold
        or features["contentOverlap"] >= strongContentOverlapThreshold
    )


def hasParallelDefinitionCrossLink(similarity: float, features: dict) -> bool:
    # Two "what is X" questions are only a parallel-definition cross-link when
    # they are actually about the same area. Matching definitional phrasing is
    # not enough — require both topic proximity and a surface anchor (shared
    # content terms, or strong topic similarity).
    if features["parallelDefinitionScore"] == 0:
        return False
    hasTopicProximity = (
        similarity >= parallelDefinitionRelatedThreshold
        or features["userTopicSimilarity"] >= strongTopicSimilarityThreshold
    )
    hasSurfaceAnchor = (
        features["contentOverlap"] > 0
        or features["userTopicSimilarity"] >= relatedSimilarityThreshold
    )
    return hasTopicProximity and hasSurfaceAnchor


def hasSubthreadTopicContinuation(graph: ConversationGraph, parentId: int, similarity: float, features: dict) -> bool:
    # A follow-up to a branch node can be a true continuation of that subthread
    # even without generic "how do I" discourse markers.
    parent = graph.turns[parentId]
    parentIsBranchTurn = any(label == "branch" for _, label, _ in parent.semanticParents)
    if not parentIsBranchTurn:
        return False

    strongTopicMatch = (
        similarity >= subthreadContinuationSimilarityThreshold
        or features["userTopicSimilarity"] >= subthreadTopicSimilarityThreshold
    )
    explicitSurfaceAnchor = (
        features["contentOverlap"] >= strongContentOverlapThreshold
        or features["lexicalOverlap"] >= subthreadLexicalOverlapThreshold
    )
    return strongTopicMatch and explicitSurfaceAnchor


def hasDiscourseSignal(features: dict) -> bool:
    return any(
        (
            features["clarificationScore"] > 0,
            features["referenceScore"] > 0,
            features["pronounReferenceScore"] > 0,
            features["forwardScore"] > 0,
            features["topicShiftScore"] > 0,
            features["comparisonScore"] > 0,
            features["parallelDefinitionScore"] > 0,
            features["followupQuestionScore"] > 0,
        )
    )

def candidateSignalStrength(similarity: float, features: dict, crossScores: dict) -> float:
    # Blend weak semantic and discourse evidence into one gate score.
    signal = 0.4 * max(similarity, 0.0)
    signal += 0.15 * features["userTopicSimilarity"]
    signal += 0.1 * features["answerSimilarity"]
    signal += 0.12 * min(
        1.0,
        features["clarificationScore"] + features["referenceScore"] + 0.5 * features["pronounReferenceScore"],
    )
    signal += 0.1 * min(1.0, features["forwardScore"] + features["followupQuestionScore"])
    signal += 0.08 * min(1.0, features["topicShiftScore"] + features["comparisonScore"])
    signal += 0.1 * max(crossScores.values())
    if hasComparisonFollowup(features):
        signal += 0.08
    if (
        features["parallelDefinitionScore"] > 0
        and similarity >= parallelDefinitionRelatedThreshold
    ):
        signal += 0.12
    return min(1.0, signal)


def allowsCandidate(similarity: float, features: dict, crossScores: dict) -> bool:
    # Reject only clearly bad matches; borderline cases can still be saved by final scoring.
    if similarity >= weakThreshold:
        return True
    if (
        features["parallelDefinitionScore"] > 0
        and similarity >= parallelDefinitionRelatedThreshold
    ):
        return True
    if hasComparisonFollowup(features):
        return True
    if similarity < minEmbeddingFloor and not hasDiscourseSignal(features):
        return False
    return (
        candidateSignalStrength(similarity, features, crossScores) >= candidateSignalThreshold
        or max(crossScores.values()) >= crossCandidateThreshold
    )


def scoreToConfidence(score: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return max(0.0, min(1.0, score / threshold))


def sigmoid(score: float) -> float:
    return float(1.0 / (1.0 + np.exp(-score)))


# Model scoring helpers

def embeddingSimilarity(leftEmbedding, rightEmbedding) -> float:
    return model.similarity(
        leftEmbedding.reshape(1, -1),
        rightEmbedding.reshape(1, -1),
    ).item()


def crossEncoderScore(userText: str, context: str) -> float:
    """
    Score how relevant a context is to the new user question.
    """
    cleanContext = context.strip()
    if not cleanContext:
        return 0.0

    rawScore = crossEncoder.predict([(userText, cleanContext)])[0]
    if np.isnan(rawScore):
        return 0.0
    safeScore = float(np.nan_to_num(rawScore, nan=0.0, posinf=20.0, neginf=-20.0))
    return sigmoid(safeScore)


def scoreCrossEncoderLabels(previousUserText: str, previousAiText: str, userText: str) -> dict:
    """
    Use label-specific cross-encoder prompts instead of sharing one relevance score.
    """
    # Each label gets a different prompt because it depends on different prior context.
    fullPreviousTurn = f"{previousUserText}\n{previousAiText}"
    branchContext = (
        "Relationship: branch follow-up that depends on the assistant's last answer.\n"
        f"Previous assistant answer: {previousAiText}"
    )
    continuationContext = (
        "Relationship: continuation of the same topic.\n"
        f"Previous turn:\n{fullPreviousTurn}"
    )
    relatedContext = (
        "Relationship: related lateral topic shift.\n"
        f"Previous user topic: {previousUserText}\n"
        f"Previous assistant answer: {previousAiText}"
    )
    return {
        "branch": round(
            0.6 * crossEncoderScore(userText, previousAiText)
            + 0.4 * crossEncoderScore(userText, branchContext),
            3,
        ),
        "continuation": round(
            0.65 * crossEncoderScore(userText, fullPreviousTurn)
            + 0.35 * crossEncoderScore(userText, continuationContext),
            3,
        ),
        "related": round(
            0.55 * crossEncoderScore(userText, previousUserText)
            + 0.45 * crossEncoderScore(userText, relatedContext),
            3,
        ),
    }


# Feature extraction

def extractDiscourseFeatures(userText: str, previousUserText: str, previousAiText: str) -> dict:
    """
    Extract discourse-level feature scores from the current and previous messages.
    
    Returns a dict with numeric scores for each feature type.
    """
    userLower = userText.lower()
    prevLower = (previousUserText + " " + previousAiText).lower()
    
    # Count pattern matches for each category
    clarificationScore = countPatternMatches(userLower, clarificationPatterns)
    referenceScore = countPatternMatches(userLower, referencePatterns)
    pronounReferenceScore = countPatternMatches(userLower, pronounReferencePatterns)
    forwardScore = countPatternMatches(userLower, forwardPatterns)
    sequencingScore = countPatternMatches(userLower, sequencingPatterns)
    topicShiftScore = countPatternMatches(userLower, topicShiftPatterns)
    comparisonScore = countPatternMatches(userLower, comparisonPatterns)
    followupQuestionScore = countPatternMatches(userLower, followupPatterns)
    parallelDefinitionScore = int(
        userLower.startswith(definitionPrefixes)
        and previousUserText.lower().startswith(definitionPrefixes)
    )
    
    # Lexical overlap analysis
    prevTerms = set(prevLower.split())
    userTerms = set(userLower.split())
    prevContentTerms = contentTerms(prevLower)
    userContentTerms = contentTerms(userLower)
    
    lexicalOverlap = 0.0
    newTermRatio = 0.0
    contentOverlap = 0.0
    if userTerms:
        lexicalOverlap = len(prevTerms & userTerms) / len(userTerms)
        newTermRatio = len(userTerms - prevTerms) / len(userTerms)
    if userContentTerms:
        contentOverlap = len(prevContentTerms & userContentTerms) / len(userContentTerms)

    # Add semantic similarities alongside phrase features so paraphrases still register.
    userEmbedding = encodeText(userText)
    previousUserEmbedding = encodeText(previousUserText)
    previousAiEmbedding = encodeText(previousAiText) if previousAiText.strip() else None
    userTopicSimilarity = embeddingSimilarity(userEmbedding, previousUserEmbedding)
    answerSimilarity = (
        embeddingSimilarity(userEmbedding, previousAiEmbedding)
        if previousAiEmbedding is not None
        else 0.0
    )
    
    features = {
        "clarificationScore": clarificationScore,
        "referenceScore": referenceScore,
        "pronounReferenceScore": pronounReferenceScore,
        "forwardScore": forwardScore,
        "sequencingScore": sequencingScore,
        "topicShiftScore": topicShiftScore,
        "comparisonScore": comparisonScore,
        "followupQuestionScore": followupQuestionScore,
        "parallelDefinitionScore": parallelDefinitionScore,
        "lexicalOverlap": lexicalOverlap,
        "newTermRatio": newTermRatio,
        "contentOverlap": contentOverlap,
        "userTopicSimilarity": userTopicSimilarity,
        "answerSimilarity": answerSimilarity,
        "isShort": len(userText.split()) <= shortMsgWords,
        "endsWithQuestion": userText.strip().endswith("?"),
    }
    
    return features


# Label scoring

def scoreEdgeLabels(similarity: float, features: dict) -> dict:
    """
    Score each possible edge label using weighted features.
    
    This is the heuristic baseline before any cross-encoder upgrade.
    
    Returns dict of scores: {"branch": score, "continuation": score, "related": score}
    """
    scores = {
        "branch": 0.0,
        "continuation": 0.0,
        "related": 0.0,
    }
    
    # Branch: Requires both clarification/reference intent AND sufficient prior context
    # Clarifications like "what do you mean?" need the previous answer to make sense
    scores["branch"] += 2.5 * features["clarificationScore"]
    scores["branch"] += 1.5 * features["referenceScore"]
    scores["branch"] += 0.5 * features["pronounReferenceScore"]
    scores["branch"] += 1.0 * features["followupQuestionScore"]
    scores["branch"] += 0.9 * features["answerSimilarity"]
    # Only penalize short messages if they lack clarification intent
    if features["isShort"] and features["clarificationScore"] == 0:
        scores["branch"] -= 0.5
    
    # Continuation: Moves the topic forward on the same thread
    # Similarity strengthens continuation, but should not create it by itself.
    hasForwardIntent = features["forwardScore"] > 0 and hasForwardAnchor(similarity, features)
    hasComparisonIntent = hasComparisonFollowup(features)
    if hasForwardIntent:
        scores["continuation"] += 2.5 * features["forwardScore"]
    if hasComparisonIntent:
        scores["continuation"] += 2.0 * features["comparisonScore"]
    if hasForwardIntent or hasComparisonIntent:
        scores["continuation"] += 2.0 * similarity
        scores["continuation"] += 0.5 * features["contentOverlap"]
    scores["continuation"] += 0.8 * features["userTopicSimilarity"]
    scores["continuation"] += 0.5 * features["answerSimilarity"]
    
    # Related: Lateral jump within same domain
    # Topic shifts with new terminology are typical
    scores["related"] += 2.0 * features["topicShiftScore"]
    scores["related"] += 0.85 * similarity
    scores["related"] += 1.0 * features["newTermRatio"]  # Introducing new concepts
    scores["related"] += 1.2 * features["parallelDefinitionScore"]
    scores["related"] += 0.9 * features["userTopicSimilarity"]
    scores["related"] += 0.7 * max(0.0, features["userTopicSimilarity"] - features["answerSimilarity"])
    
    return scores


def judgeEdgeLabel(
    similarity: float,
    features: dict,
    labelScores: dict,
    crossScores: dict,
) -> dict:
    """
    Convert heuristic label scores plus cross-encoder evidence into label confidences.
    """
    # Only compute confidence for labels that have some supporting signal.
    # The pronoun-reference branch uses _maxTopicalSignal (any of similarity /
    # userTopicSimilarity / answerSimilarity / contentOverlap), not
    # answerSimilarity alone: a real pronoun reference ("that", "it") to the
    # prior turn is strong discourse evidence on its own, so it only needs
    # topic identity to not be completely absent, the same minEmbeddingFloor
    # bar the embedding gate itself uses elsewhere — not a specific 0.3 on one
    # particular signal, which a verbose or tangential real answer can easily
    # miss despite the reference being unambiguous.
    hasDependencySignal = (
        features["clarificationScore"] > 0
        or features["referenceScore"] > 0
        or (
            features["pronounReferenceScore"] > 0
            and _maxTopicalSignal(similarity, features) >= minEmbeddingFloor
        )
        or (
            features["followupQuestionScore"] > 0
            and features["answerSimilarity"] >= parallelDefinitionRelatedThreshold
        )
    )
    hasContinuationSignal = (
        hasForwardAnchor(similarity, features) and features["forwardScore"] > 0
        or hasComparisonFollowup(features)
        or features["comparisonScore"] > 0
        or (
            features["userTopicSimilarity"] >= THRESHOLD
            and features["answerSimilarity"] >= parallelDefinitionRelatedThreshold
        )
    )
    hasRelatedSignal = (
        features["topicShiftScore"] > 0
        or features["parallelDefinitionScore"] > 0
        or labelScores["related"] >= relatedMinScore
        or (
            features["userTopicSimilarity"] >= parallelDefinitionRelatedThreshold
            and features["newTermRatio"] >= 0.5
        )
    )

    confidences = {
        "branch": 0.0,
        "continuation": 0.0,
        "related": 0.0,
        "unrelated": 0.0,
    }

    if hasDependencySignal:
        heuristicConfidence = scoreToConfidence(labelScores["branch"], branchMinScore)
        confidences["branch"] = (
            0.55 * heuristicConfidence
            + 0.25 * crossScores["branch"]
            + 0.15 * features["answerSimilarity"]
            + 0.05 * min(
                1.0,
                features["referenceScore"] + features["clarificationScore"] + 0.5 * features["pronounReferenceScore"],
            )
        )

    if hasContinuationSignal:
        heuristicConfidence = scoreToConfidence(labelScores["continuation"], continuationMinScore)
        confidences["continuation"] = (
            0.55 * heuristicConfidence
            + 0.2 * crossScores["continuation"]
            + 0.1 * similarity
            + 0.1 * features["userTopicSimilarity"]
            + 0.05 * min(1.0, features["forwardScore"] + features["comparisonScore"])
        )

    if (
        hasRelatedSignal
        and (
            similarity >= relatedSimilarityThreshold
            or (
                features["parallelDefinitionScore"] > 0
                and similarity >= parallelDefinitionRelatedThreshold
            )
            or (
                # A "what about X instead"-style lateral shift is strong
                # discourse evidence by itself; same reasoning as the
                # pronoun-reference case in hasDependencySignal above — any
                # topical signal at all (not userTopicSimilarity specifically)
                # clearing the shared minEmbeddingFloor is enough, since a
                # verbose real answer can suppress one signal while another
                # still shows the lateral shift is topically grounded.
                features["topicShiftScore"] > 0
                and _maxTopicalSignal(similarity, features) >= minEmbeddingFloor
            )
        )
    ):
        heuristicConfidence = scoreToConfidence(labelScores["related"], relatedMinScore)
        confidences["related"] = (
            0.5 * heuristicConfidence
            + 0.15 * crossScores["related"]
            + 0.15 * max(similarity, weakThreshold)
            + 0.15 * features["userTopicSimilarity"]
            + 0.05 * min(1.0, features["topicShiftScore"] + features["parallelDefinitionScore"])
        )
        if (
            features["parallelDefinitionScore"] > 0
            and similarity >= parallelDefinitionRelatedThreshold
        ):
            confidences["related"] = max(confidences["related"], edgeConfidenceThreshold)

    strongestEdgeConfidenceValue = max(
        confidences["branch"],
        confidences["continuation"],
        confidences["related"],
    )
    confidences["unrelated"] = round(max(0.0, 1.0 - strongestEdgeConfidenceValue), 3)

    return {label: round(confidence, 3) for label, confidence in confidences.items()}


def selectEdgeLabel(confidences: dict) -> tuple[str, float] | None:
    # confidences["unrelated"] is always exactly 1 - bestConfidence (see
    # judgeEdgeLabel), so "unrelated >= best" is definitionally just another
    # way of writing "best <= 0.5" — a second, hardcoded 0.5 floor that used
    # to silently reintroduce the old cutoff no matter what
    # edgeConfidenceThreshold was actually set to. edgeConfidenceThreshold
    # below is the one real "confident enough to commit" bar.
    edgeConfidences = {
        label: confidence
        for label, confidence in confidences.items()
        if label != "unrelated"
    }
    bestLabel = max(edgeConfidences, key=edgeConfidences.get)
    bestConfidence = edgeConfidences[bestLabel]

    if bestConfidence < edgeConfidenceThreshold:
        return None

    return bestLabel, bestConfidence


def strongestEdgeConfidence(confidences: dict) -> float:
    return max(
        confidences["branch"],
        confidences["continuation"],
        confidences["related"],
    )


# Cross-link retrieval

def retrieveCrossLinkCandidates(graph: ConversationGraph, embedding, timelineParent: int | None) -> list[tuple[int, float]]:
    # Direct top-k retrieval: score every prior turn by cosine similarity to the
    # new turn, with a small decay for older turns. Cheap at this scale and
    # avoids the concept-centroid layer, whose average embedding is a poor
    # representative once a concept has drifted.
    scored = []
    for turn in graph.turns:
        if turn.id == timelineParent or turn.embedding is None:
            continue
        similarity = embeddingSimilarity(embedding, turn.embedding)
        agePenalty = olderTurnDecay * max(0, len(graph.turns) - 1 - turn.id)
        scored.append((turn.id, similarity - agePenalty))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:maxOlderCandidates]


# Relationship analyzers

def analyzeImmediateRelationship(
    previousUserText: str,
    previousAiText: str,
    userText: str,
    similarity: float | None = None,
) -> dict:
    if similarity is None:
        previousEmbedding = model.encode(previousUserText + " " + previousAiText)
        userEmbedding = model.encode(userText)
        similarity = model.similarity(
            userEmbedding.reshape(1, -1),
            previousEmbedding.reshape(1, -1),
        ).item()

    features = extractDiscourseFeatures(userText, previousUserText, previousAiText)
    crossScores = scoreCrossEncoderLabels(previousUserText, previousAiText, userText)
    labelScores = scoreEdgeLabels(similarity, features)
    confidences = judgeEdgeLabel(similarity, features, labelScores, crossScores)
    candidateAllowed = allowsCandidate(similarity, features, crossScores)
    if not candidateAllowed and strongestEdgeConfidence(confidences) < edgeConfidenceThreshold:
        selectedEdge = None
    else:
        selectedEdge = selectEdgeLabel(confidences)

    return {
        "candidateAllowed": candidateAllowed,
        "candidateSignalStrength": round(candidateSignalStrength(similarity, features, crossScores), 3),
        "similarity": round(similarity, 3),
        "features": features,
        "heuristicScores": {label: round(score, 3) for label, score in labelScores.items()},
        "crossEncoderScores": crossScores,
        "confidences": confidences,
        "selectedLabel": selectedEdge[0] if selectedEdge is not None else "unrelated",
        "selectedConfidence": selectedEdge[1] if selectedEdge is not None else confidences["unrelated"],
    }


def classifySemanticCrossLink(graph: ConversationGraph, parentId: int, userText: str, similarity: float) -> tuple[str, float]:
    # Older links only choose between continuation and related.
    parent = graph.turns[parentId]
    features = extractDiscourseFeatures(userText, parent.userText, parent.aiText)
    crossScores = scoreCrossEncoderLabels(parent.userText, parent.aiText, userText)
    labelScores = scoreEdgeLabels(similarity, features)
    hasForwardContinuation = features["forwardScore"] > 0 and hasForwardAnchor(similarity, features)
    hasSubthreadContinuation = hasSubthreadTopicContinuation(graph, parentId, similarity, features)
    if hasComparisonFollowup(features) or (
        hasForwardContinuation and labelScores["continuation"] >= labelScores["related"]
    ) or hasSubthreadContinuation:
        confidence = (
            0.6 * scoreToConfidence(labelScores["continuation"], continuationMinScore)
            + 0.4 * crossScores["continuation"]
            + (0.08 if hasSubthreadContinuation else 0.0)
        )
        return "continuation", round(max(confidence, edgeConfidenceThreshold), 3)
    relatedConfidence = max(
        edgeConfidenceThreshold,
        0.65 * similarity + 0.2 * features["userTopicSimilarity"] + 0.15 * crossScores["related"],
    )
    return "related", round(relatedConfidence, 3)


def immediatePreviousTurnClassifier(
    graph: ConversationGraph,
    embedding,
    timelineParentId: int,
    userText: str
) -> tuple[int, str, float] | None:
    """
    3-layer classifier for the immediate previous turn.
    
    Layer 1: Embedding similarity (candidate gating)
    Layer 2: Discourse feature scoring (heuristic baseline)
    Layer 3: Cross-encoder final relationship confidence
    
    Returns (parentId, "continuation" | "branch" | "related", confidence) or None.
    """
    # The immediate previous turn gets the full label set and strongest classifier.
    parent = graph.turns[timelineParentId]

    # Layer 1: Embedding similarity (candidate filter)
    scoreToParent = embeddingSimilarity(embedding, parent.embedding)

    features = extractDiscourseFeatures(userText, parent.userText, parent.aiText)
    crossScores = scoreCrossEncoderLabels(parent.userText, parent.aiText, userText)
    
    # Layer 2: Discourse feature scoring (heuristic baseline)
    labelScores = scoreEdgeLabels(scoreToParent, features)
    
    # Layer 3: cross-encoder label-specific confidence
    confidences = judgeEdgeLabel(scoreToParent, features, labelScores, crossScores)
    candidateAllowed = allowsCandidate(scoreToParent, features, crossScores)
    if not candidateAllowed and strongestEdgeConfidence(confidences) < edgeConfidenceThreshold:
        return None
    judgedLabel = selectEdgeLabel(confidences)
    if judgedLabel is not None:
        label, confidence = judgedLabel
        return (timelineParentId, label, confidence)
    
    return None


def isSemanticAncestor(graph: ConversationGraph, descendantId: int, ancestorId: int) -> bool:
    stack = [descendantId]
    seen = set()

    while stack:
        currentId = stack.pop()
        if currentId in seen or currentId >= len(graph.turns):
            continue
        seen.add(currentId)
        for parentId, _, _ in graph.turns[currentId].semanticParents:
            if parentId == ancestorId:
                return True
            stack.append(parentId)

    return False


def pruneLessSpecificSemanticResults(graph: ConversationGraph, semanticResults: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    prunedResults: list[tuple[int, str, float]] = []

    for result in semanticResults:
        parentId, label, confidence = result
        overshadowed = False
        for otherParentId, otherLabel, otherConfidence in semanticResults:
            if otherParentId == parentId:
                continue
            if not isSemanticAncestor(graph, otherParentId, parentId):
                continue

            otherIsMoreSpecific = otherLabel == "continuation" and otherConfidence >= confidence - 0.06
            if label == "related" and otherIsMoreSpecific:
                overshadowed = True
                break

        if not overshadowed:
            prunedResults.append(result)

    return prunedResults


def findSemanticParents(graph: ConversationGraph, embedding, timelineParent: int | None, userText: str) -> list[tuple[int, str, float]]:
    """
    Find all turns the new turn should attach to in the graph.

    Step 1: classify against the immediately preceding turn using discourse analysis.
    Step 2: scan all previous turns for other high-similarity matches (cross-thread related).

    Returns a list of (turnId, "continuation" | "branch" | "related", confidence) edges.
    """
    if not graph.turns:
        return []

    # Step 1 handles the timeline parent; step 2 looks for older semantic cross-links.
    timelineResult = immediatePreviousTurnClassifier(graph, embedding, timelineParent, userText) if timelineParent is not None else None

    # step 2: retrieve candidate concepts, then score only their member turns
    scored = retrieveCrossLinkCandidates(graph, embedding, timelineParent)
    if not scored:
        return [timelineResult] if timelineResult is not None else []

    bestScore = scored[0][1]
    semanticResults = []
    hasComparisonQuestion = countPatternMatches(userText.lower(), comparisonPatterns) > 0

    # Discourse features are reused by the parallel-definition gate and the
    # classification loop below, so compute them once per candidate.
    candidateFeatures = {
        id: extractDiscourseFeatures(userText, graph.turns[id].userText, graph.turns[id].aiText)
        for id, _ in scored
    }

    hasParallelDefinitionCandidate = any(
        hasParallelDefinitionCrossLink(score, candidateFeatures[id])
        for id, score in scored
    )

    if bestScore >= THRESHOLD or hasComparisonQuestion or hasParallelDefinitionCandidate:
        for id, score in scored:
            if len(semanticResults) >= maxParents:
                break
            crossFeatures = candidateFeatures[id]
            if hasComparisonFollowup(crossFeatures):
                label, confidence = classifySemanticCrossLink(graph, id, userText, score)
                semanticResults.append((id, label, confidence))
            elif hasParallelDefinitionCrossLink(score, crossFeatures):
                label, confidence = classifySemanticCrossLink(graph, id, userText, score)
                semanticResults.append((id, label, confidence))
            elif score >= bestScore - closeGap:
                # semantically close but not the timeline parent → classify cross-link
                label, confidence = classifySemanticCrossLink(graph, id, userText, score)
                semanticResults.append((id, label, confidence))

    semanticResults = pruneLessSpecificSemanticResults(graph, semanticResults)

    if (
        timelineResult is not None
        and timelineParent is not None
        and timelineResult[1] == "continuation"
    ):
        timelineConfidence = timelineResult[2]
        strongerAncestorContinuation = next(
            (
                result
                for result in semanticResults
                if result[1] == "continuation"
                and isSemanticAncestor(graph, timelineParent, result[0])
                and result[2] >= timelineConfidence - 0.05
            ),
            None,
        )
        if strongerAncestorContinuation is not None:
            timelineResult = None

    # timeline result goes first, related cross-links after
    if timelineResult is not None:
        return [timelineResult] + semanticResults
    return semanticResults


def addTurn(graph: ConversationGraph, userText: str, aiText: str) -> TurnModel:
    """
    Create a new turn from a user message and AI response, attach it to the graph,
    and assign it to the appropriate concept(s).
    """
    # the previous turn in the timeline (always the last one added)
    timelineParent = graph.turns[-1].id if graph.turns else None

    # use the new user question to classify its relationship to existing turns
    userEmbedding = encodeText(userText)

    # store the full turn embedding for future semantic context
    combined = userText + " " + aiText
    embedding = encodeText(combined)

    # find which previous turns this turn relates to, and how
    semanticParents = findSemanticParents(graph, userEmbedding, timelineParent, userText)

    # no semantic parents → this turn starts a new topic
    root = len(semanticParents) == 0

    if root:
        # new concept: assign the next available concept id
        conceptIds = [graph.conceptCounter]
        graph.conceptCounter += 1
    else:
        # inherit concept ids from all semantic parents (deduplicated)
        # if parents span two concepts, this turn belongs to both
        seen = set()
        conceptIds = []
        for parentId, _, _ in semanticParents:
            for cid in graph.turns[parentId].conceptIds:
                if cid not in seen:
                    seen.add(cid)
                    conceptIds.append(cid)

    turn = TurnModel(
        id=nextId(graph),
        root=root,
        embedding=embedding,
        userText=userText,
        aiText=aiText,
        timestamp=datetime.now(),
        timelineParent=timelineParent,
        semanticParents=semanticParents,
        conceptIds=conceptIds,
    )
    graph.turns.append(turn)
    return turn


def resetConversation(graph: ConversationGraph) -> None:
    graph.turns.clear()
    graph.conceptCounter = 0


def reclassifyTurns(graph: ConversationGraph) -> list[TurnModel]:
    existingTurns = [
        (turn.userText, turn.aiText, turn.timestamp)
        for turn in graph.turns
    ]

    resetConversation(graph)

    rebuiltTurns = []
    for userText, aiText, timestamp in existingTurns:
        turn = addTurn(graph, userText, aiText)
        turn.timestamp = timestamp
        rebuiltTurns.append(turn)

    return rebuiltTurns


def loadConversationGraph(conversationId: int) -> ConversationGraph:
    graph = ConversationGraph()
    storedTurns = getConversationTurns(conversationId)

    for storedTurn in storedTurns:
        # Stored as a float32 BLOB; copy so the array owns its buffer.
        embedding = (
            np.frombuffer(storedTurn["embedding"], dtype=np.float32).copy()
            if storedTurn["embedding"] is not None
            else None
        )
        turn = TurnModel(
            id=storedTurn["id"],
            root=storedTurn["root"],
            embedding=embedding,
            userText=storedTurn["userText"],
            aiText=storedTurn["aiText"],
            timestamp=datetime.fromisoformat(storedTurn["timestamp"]),
            timelineParent=storedTurn["timelineParent"],
            semanticParents=storedTurn["semanticParents"],
            conceptIds=storedTurn["conceptIds"],
        )
        graph.turns.append(turn)

    maxConceptId = max((conceptId for turn in graph.turns for conceptId in turn.conceptIds), default=-1)
    graph.conceptCounter = maxConceptId + 1
    return graph
