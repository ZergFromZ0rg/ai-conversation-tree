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
turns: list[TurnModel] = []
conceptMembers: dict[int, list[int]] = {}
conceptEmbeddingSums: dict[int, np.ndarray] = {}
conceptTurnCounts: dict[int, int] = {}

conceptCounter = 0

THRESHOLD = 0.55
continuationThreshold = 0.55
closeGap = 0.05
weakThreshold = 0.35
minEmbeddingFloor = 0.12
branchMinScore = 1.0
continuationMinScore = 1.2
relatedMinScore = 2.25
relatedSimilarityThreshold = THRESHOLD
edgeConfidenceThreshold = 0.5
candidateSignalThreshold = 0.52
crossCandidateThreshold = 0.5
parallelDefinitionRelatedThreshold = 0.3
maxParents = 2
shortMsgWords = 5
topConcepts = 3
maxCandidatesPerConcept = 3
maxOlderCandidates = 5
conceptScoreMargin = 0.08
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

def nextId():
    return len(turns)


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


def hasForwardAnchor(similarity: float, features: dict) -> bool:
    # Generic "how do I learn/use X" phrasing should only count as continuation
    # when topic identity is clearly established, not from incidental overlap.
    return (
        similarity >= strongTopicSimilarityThreshold
        or features["userTopicSimilarity"] >= strongTopicSimilarityThreshold
        or features["answerSimilarity"] >= strongTopicSimilarityThreshold
        or features["contentOverlap"] >= strongContentOverlapThreshold
    )


def hasParallelDefinitionCrossLink(similarity: float, features: dict) -> bool:
    return (
        features["parallelDefinitionScore"] > 0
        and (
            similarity >= minEmbeddingFloor
            or features["userTopicSimilarity"] >= strongTopicSimilarityThreshold
        )
    )


def hasSubthreadTopicContinuation(parentId: int, similarity: float, features: dict) -> bool:
    # A follow-up to a branch node can be a true continuation of that subthread
    # even without generic "how do I" discourse markers.
    parent = turns[parentId]
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
    hasDependencySignal = (
        features["clarificationScore"] > 0
        or features["referenceScore"] > 0
        or (
            features["pronounReferenceScore"] > 0
            and features["answerSimilarity"] >= parallelDefinitionRelatedThreshold
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
                features["topicShiftScore"] > 0
                and features["userTopicSimilarity"] >= 0.18
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
    edgeConfidences = {
        label: confidence
        for label, confidence in confidences.items()
        if label != "unrelated"
    }
    bestLabel = max(edgeConfidences, key=edgeConfidences.get)
    bestConfidence = edgeConfidences[bestLabel]

    if confidences["unrelated"] >= bestConfidence:
        return None
    if bestConfidence < edgeConfidenceThreshold:
        return None

    return bestLabel, bestConfidence


def strongestEdgeConfidence(confidences: dict) -> float:
    return max(
        confidences["branch"],
        confidences["continuation"],
        confidences["related"],
    )


# Concept index helpers

def registerTurnInConcepts(turn: TurnModel):
    # Update the per-concept cache incrementally instead of rebuilding it from all turns.
    for conceptId in turn.conceptIds:
        conceptMembers.setdefault(conceptId, []).append(turn.id)
        if conceptId in conceptEmbeddingSums:
            conceptEmbeddingSums[conceptId] = conceptEmbeddingSums[conceptId] + turn.embedding
            conceptTurnCounts[conceptId] += 1
        else:
            conceptEmbeddingSums[conceptId] = np.array(turn.embedding, copy=True)
            conceptTurnCounts[conceptId] = 1


def conceptTurnIds(conceptId: int) -> list[int]:
    return conceptMembers.get(conceptId, [])


def conceptCentroid(conceptId: int):
    # The centroid is the running average embedding for one concept.
    turnCount = conceptTurnCounts.get(conceptId, 0)
    if turnCount == 0:
        return None
    return conceptEmbeddingSums[conceptId] / turnCount


def conceptSimilarityScores(embedding) -> list[tuple[int, float]]:
    scored = []
    for conceptId in sorted(conceptTurnCounts):
        centroid = conceptCentroid(conceptId)
        score = embeddingSimilarity(embedding, centroid)
        scored.append((conceptId, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def retrieveCrossLinkCandidates(embedding, timelineParent: int | None) -> list[tuple[int, float]]:
    # Retrieve older candidates in two stages: top concepts first, then top turns inside them.
    conceptScores = conceptSimilarityScores(embedding)
    if not conceptScores:
        return []

    topConceptScore = conceptScores[0][1]
    selectedConcepts = [
        (conceptId, score)
        for conceptId, score in conceptScores[:topConcepts]
        if score >= topConceptScore - conceptScoreMargin
    ]

    scoredCandidates = {}
    for conceptId, conceptScore in selectedConcepts:
        memberIds = [
            turnId for turnId in conceptTurnIds(conceptId)
            if turnId != timelineParent
        ]
        if not memberIds:
            continue
        conceptTurnScores = []
        for turnId in memberIds:
            similarity = embeddingSimilarity(embedding, turns[turnId].embedding)
            agePenalty = olderTurnDecay * max(0, len(turns) - 1 - turnId)
            adjustedScore = similarity - agePenalty + 0.15 * conceptScore
            conceptTurnScores.append((turnId, adjustedScore))

        conceptTurnScores.sort(key=lambda item: item[1], reverse=True)
        for turnId, adjustedScore in conceptTurnScores[:maxCandidatesPerConcept]:
            existing = scoredCandidates.get(turnId)
            if existing is None or adjustedScore > existing:
                scoredCandidates[turnId] = adjustedScore

    return sorted(
        scoredCandidates.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:maxOlderCandidates]


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


def classifySemanticCrossLink(parentId: int, userText: str, similarity: float) -> tuple[str, float]:
    # Older links only choose between continuation and related.
    parent = turns[parentId]
    features = extractDiscourseFeatures(userText, parent.userText, parent.aiText)
    crossScores = scoreCrossEncoderLabels(parent.userText, parent.aiText, userText)
    labelScores = scoreEdgeLabels(similarity, features)
    hasForwardContinuation = features["forwardScore"] > 0 and hasForwardAnchor(similarity, features)
    hasSubthreadContinuation = hasSubthreadTopicContinuation(parentId, similarity, features)
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
    parent = turns[timelineParentId]
    
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


def isSemanticAncestor(descendantId: int, ancestorId: int) -> bool:
    stack = [descendantId]
    seen = set()

    while stack:
        currentId = stack.pop()
        if currentId in seen or currentId >= len(turns):
            continue
        seen.add(currentId)
        for parentId, _, _ in turns[currentId].semanticParents:
            if parentId == ancestorId:
                return True
            stack.append(parentId)

    return False


def pruneLessSpecificSemanticResults(semanticResults: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    prunedResults: list[tuple[int, str, float]] = []

    for result in semanticResults:
        parentId, label, confidence = result
        overshadowed = False
        for otherParentId, otherLabel, otherConfidence in semanticResults:
            if otherParentId == parentId:
                continue
            if not isSemanticAncestor(otherParentId, parentId):
                continue

            otherIsMoreSpecific = otherLabel == "continuation" and otherConfidence >= confidence - 0.06
            if label == "related" and otherIsMoreSpecific:
                overshadowed = True
                break

        if not overshadowed:
            prunedResults.append(result)

    return prunedResults


def findSemanticParents(embedding, timelineParent: int | None, userText: str) -> list[tuple[int, str, float]]:
    """
    Find all turns the new turn should attach to in the graph.

    Step 1: classify against the immediately preceding turn using discourse analysis.
    Step 2: scan all previous turns for other high-similarity matches (cross-thread related).

    Returns a list of (turnId, "continuation" | "branch" | "related", confidence) edges.
    """
    if not turns:
        return []

    # Step 1 handles the timeline parent; step 2 looks for older semantic cross-links.
    timelineResult = immediatePreviousTurnClassifier(embedding, timelineParent, userText) if timelineParent is not None else None

    # step 2: retrieve candidate concepts, then score only their member turns
    scored = retrieveCrossLinkCandidates(embedding, timelineParent)
    if not scored:
        return [timelineResult] if timelineResult is not None else []

    bestScore = scored[0][1]
    semanticResults = []
    hasComparisonQuestion = countPatternMatches(userText.lower(), comparisonPatterns) > 0

    hasParallelDefinitionCandidate = any(
        hasParallelDefinitionCrossLink(
            score,
            extractDiscourseFeatures(userText, turns[id].userText, turns[id].aiText),
        )
        for id, score in scored[:maxOlderCandidates]
    )

    if bestScore >= THRESHOLD or hasComparisonQuestion or hasParallelDefinitionCandidate:
        for id, score in scored:
            if len(semanticResults) >= maxParents:
                break
            crossFeatures = extractDiscourseFeatures(userText, turns[id].userText, turns[id].aiText)
            if hasComparisonFollowup(crossFeatures):
                label, confidence = classifySemanticCrossLink(id, userText, score)
                semanticResults.append((id, label, confidence))
            elif hasParallelDefinitionCrossLink(score, crossFeatures):
                label, confidence = classifySemanticCrossLink(id, userText, score)
                semanticResults.append((id, label, confidence))
            elif score >= bestScore - closeGap:
                # semantically close but not the timeline parent → classify cross-link
                label, confidence = classifySemanticCrossLink(id, userText, score)
                semanticResults.append((id, label, confidence))

    semanticResults = pruneLessSpecificSemanticResults(semanticResults)

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
                and isSemanticAncestor(timelineParent, result[0])
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


def addTurn(userText: str, aiText: str) -> TurnModel:
    """
    Create a new turn from a user message and AI response, attach it to the graph,
    and assign it to the appropriate concept(s).
    """
    global conceptCounter

    # the previous turn in the timeline (always the last one added)
    timelineParent = turns[-1].id if turns else None

    # use the new user question to classify its relationship to existing turns
    userEmbedding = encodeText(userText)

    # store the full turn embedding for future semantic context
    combined = userText + " " + aiText
    embedding = encodeText(combined)

    # find which previous turns this turn relates to, and how
    semanticParents = findSemanticParents(userEmbedding, timelineParent, userText)

    # no semantic parents → this turn starts a new topic
    root = len(semanticParents) == 0

    if root:
        # new concept: assign the next available concept id
        conceptIds = [conceptCounter]
        conceptCounter += 1
    else:
        # inherit concept ids from all semantic parents (deduplicated)
        # if parents span two concepts, this turn belongs to both
        seen = set()
        conceptIds = []
        for parentId, _, _ in semanticParents:
            for cid in turns[parentId].conceptIds:
                if cid not in seen:
                    seen.add(cid)
                    conceptIds.append(cid)

    turn = TurnModel(
        id=nextId(),
        root=root,
        embedding=embedding,
        userText=userText,
        aiText=aiText,
        timestamp=datetime.now(),
        timelineParent=timelineParent,
        semanticParents=semanticParents,
        conceptIds=conceptIds,
    )
    turns.append(turn)
    registerTurnInConcepts(turn)
    return turn


def resetConversation():
    global conceptCounter

    turns.clear()
    conceptMembers.clear()
    conceptEmbeddingSums.clear()
    conceptTurnCounts.clear()
    conceptCounter = 0


def reclassifyTurns() -> list[TurnModel]:
    existingTurns = [
        (turn.userText, turn.aiText, turn.timestamp)
        for turn in turns
    ]

    resetConversation()

    rebuiltTurns = []
    for userText, aiText, timestamp in existingTurns:
        turn = addTurn(userText, aiText)
        turn.timestamp = timestamp
        rebuiltTurns.append(turn)

    return rebuiltTurns


def loadConversationState(conversationId: int):
    resetConversation()
    storedTurns = getConversationTurns(conversationId)

    for storedTurn in storedTurns:
        # Persisted JSON embeddings reload as Python floats, so force float32 to
        # match fresh encoder outputs and keep similarity math consistent.
        embedding = np.array(storedTurn["embedding"], dtype=np.float32) if storedTurn["embedding"] is not None else None
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
        turns.append(turn)
        registerTurnInConcepts(turn)

    global conceptCounter
    maxConceptId = max((conceptId for turn in turns for conceptId in turn.conceptIds), default=-1)
    conceptCounter = maxConceptId + 1
