from datetime import datetime
from tree import Turn

import numpy as np
import re
from sentence_transformers import CrossEncoder, SentenceTransformer

BI_ENCODER_MODEL_NAME = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

model = SentenceTransformer(BI_ENCODER_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
turns: list[Turn] = []

concept_counter = 0

THRESHOLD = 0.55
CONTINUATION_THRESHOLD = 0.55
CLOSE_GAP = 0.05
WEAK_THRESHOLD = 0.35
BRANCH_MIN_SCORE = 1.0
CONTINUATION_MIN_SCORE = 1.2
RELATED_MIN_SCORE = 2.25
RELATED_SIMILARITY_THRESHOLD = THRESHOLD
EDGE_CONFIDENCE_THRESHOLD = 0.55
PARALLEL_DEFINITION_RELATED_THRESHOLD = 0.3
MAX_PARENTS = 2
SHORT_MSG_WORDS = 5

# Discourse markers for classification (improved patterns)
CLARIFICATION_PATTERNS = [
    "what do you mean",
    "can you clarify",
    "explain",
    "how so",
    "why",
    "does that mean",
]

REFERENCE_PATTERNS = [
    "that",
    "this",
    "it",
    "you said",
    "you mentioned",
    "above",
    "previous",
]

FORWARD_PATTERNS = [
    "how do i",
    "how do we",
    "implement",
    "build",
    "use",
    "next",
    "what's next",
    "should i",
    "best way",
    "tell me more",
    "more about",
    "go deeper",
    "continue",
]

TOPIC_SHIFT_PATTERNS = [
    "what about",
    "similar to",
    "like",
    "alternative",
]

COMPARISON_PATTERNS = [
    "difference between",
    "differences between",
    "compare",
    "compared to",
    "compared with",
    "vs",
    "versus",
]

DEFINITION_PREFIXES = (
    "what is ",
    "what are ",
)

STOPWORDS = {
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


def next_id():
    return len(turns)


def normalize_word(word: str) -> str:
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def content_terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {normalize_word(word) for word in words if word not in STOPWORDS}


def has_comparison_with_prior_topic(features: dict) -> bool:
    return features["comparison_score"] > 0 and features["content_overlap"] > 0


def allows_candidate(similarity: float, features: dict) -> bool:
    return (
        similarity >= WEAK_THRESHOLD
        or (
            features["parallel_definition_score"] > 0
            and similarity >= PARALLEL_DEFINITION_RELATED_THRESHOLD
        )
        or has_comparison_with_prior_topic(features)
    )


def score_to_confidence(score: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return max(0.0, min(1.0, score / threshold))


def sigmoid(score: float) -> float:
    return float(1.0 / (1.0 + np.exp(-score)))


def cross_encoder_score(user_text: str, context: str) -> float:
    """
    Score how relevant a context is to the new user question.
    """
    clean_context = context.strip()
    if not clean_context:
        return 0.0

    raw_score = cross_encoder.predict([(user_text, clean_context)])[0]
    return sigmoid(float(raw_score))


def score_cross_encoder_labels(previous_user_text: str, previous_ai_text: str, user_text: str) -> dict:
    """
    Use label-specific cross-encoder inputs instead of sharing one relevance score.
    """
    full_previous_turn = f"{previous_user_text}\n{previous_ai_text}"
    return {
        "branch": cross_encoder_score(user_text, previous_ai_text),
        "continuation": cross_encoder_score(user_text, full_previous_turn),
        "related": cross_encoder_score(user_text, previous_user_text),
    }


def extract_discourse_features(user_text: str, previous_user_text: str, previous_ai_text: str) -> dict:
    """
    Extract discourse-level feature scores from the current and previous messages.
    
    Returns a dict with numeric scores for each feature type.
    """
    user_lower = user_text.lower()
    prev_lower = (previous_user_text + " " + previous_ai_text).lower()
    
    # Count pattern matches for each category
    clarification_score = sum(1 for p in CLARIFICATION_PATTERNS if p in user_lower)
    reference_score = sum(1 for p in REFERENCE_PATTERNS if p in user_lower)
    forward_score = sum(1 for p in FORWARD_PATTERNS if p in user_lower)
    topic_shift_score = sum(1 for p in TOPIC_SHIFT_PATTERNS if p in user_lower)
    comparison_score = sum(1 for p in COMPARISON_PATTERNS if p in user_lower)
    parallel_definition_score = int(
        user_lower.startswith(DEFINITION_PREFIXES)
        and previous_user_text.lower().startswith(DEFINITION_PREFIXES)
    )
    
    # Lexical overlap analysis
    prev_terms = set(prev_lower.split())
    user_terms = set(user_lower.split())
    prev_content_terms = content_terms(prev_lower)
    user_content_terms = content_terms(user_lower)
    
    lexical_overlap = 0.0
    new_term_ratio = 0.0
    content_overlap = 0.0
    if user_terms:
        lexical_overlap = len(prev_terms & user_terms) / len(user_terms)
        new_term_ratio = len(user_terms - prev_terms) / len(user_terms)
    if user_content_terms:
        content_overlap = len(prev_content_terms & user_content_terms) / len(user_content_terms)
    
    features = {
        "clarification_score": clarification_score,
        "reference_score": reference_score,
        "forward_score": forward_score,
        "topic_shift_score": topic_shift_score,
        "comparison_score": comparison_score,
        "parallel_definition_score": parallel_definition_score,
        "lexical_overlap": lexical_overlap,
        "new_term_ratio": new_term_ratio,
        "content_overlap": content_overlap,
        "is_short": len(user_text.split()) <= SHORT_MSG_WORDS,
    }
    
    return features


def score_edge_labels(similarity: float, features: dict) -> dict:
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
    scores["branch"] += 2.5 * features["clarification_score"]
    scores["branch"] += 1.5 * features["reference_score"]
    # Only penalize short messages if they lack clarification intent
    if features["is_short"] and features["clarification_score"] == 0:
        scores["branch"] -= 0.5
    
    # Continuation: Moves the topic forward on the same thread
    # Similarity strengthens continuation, but should not create it by itself.
    has_forward_intent = features["forward_score"] > 0
    has_comparison_intent = has_comparison_with_prior_topic(features)
    if has_forward_intent:
        scores["continuation"] += 2.5 * features["forward_score"]
    if has_comparison_intent:
        scores["continuation"] += 2.0 * features["comparison_score"]
    if has_forward_intent or has_comparison_intent:
        scores["continuation"] += 2.0 * similarity
        scores["continuation"] += 0.5 * features["content_overlap"]
    
    # Related: Lateral jump within same domain
    # Topic shifts with new terminology are typical
    scores["related"] += 2.0 * features["topic_shift_score"]
    scores["related"] += 1.0 * similarity
    scores["related"] += 1.5 * features["new_term_ratio"]  # Introducing new concepts
    scores["related"] += 1.2 * features["parallel_definition_score"]
    
    return scores


def judge_edge_label(
    similarity: float,
    features: dict,
    label_scores: dict,
    cross_scores: dict,
) -> dict:
    """
    Convert heuristic label scores plus cross-encoder evidence into label confidences.
    """
    has_dependency_signal = (
        features["clarification_score"] > 0
        or features["reference_score"] > 0
    )
    has_continuation_signal = (
        features["forward_score"] > 0
        or has_comparison_with_prior_topic(features)
    )
    has_related_signal = (
        features["topic_shift_score"] > 0
        or features["parallel_definition_score"] > 0
        or label_scores["related"] >= RELATED_MIN_SCORE
    )

    confidences = {
        "branch": 0.0,
        "continuation": 0.0,
        "related": 0.0,
        "unrelated": 0.0,
    }

    if has_dependency_signal:
        heuristic_confidence = score_to_confidence(label_scores["branch"], BRANCH_MIN_SCORE)
        confidences["branch"] = (
            0.55 * heuristic_confidence
            + 0.45 * cross_scores["branch"]
        )

    if has_continuation_signal:
        heuristic_confidence = score_to_confidence(label_scores["continuation"], CONTINUATION_MIN_SCORE)
        confidences["continuation"] = (
            0.45 * heuristic_confidence
            + 0.4 * cross_scores["continuation"]
            + 0.15 * similarity
        )

    if (
        has_related_signal
        and (
            similarity >= RELATED_SIMILARITY_THRESHOLD
            or (
                features["parallel_definition_score"] > 0
                and similarity >= PARALLEL_DEFINITION_RELATED_THRESHOLD
            )
        )
    ):
        heuristic_confidence = score_to_confidence(label_scores["related"], RELATED_MIN_SCORE)
        confidences["related"] = (
            0.45 * heuristic_confidence
            + 0.35 * cross_scores["related"]
            + 0.2 * max(similarity, WEAK_THRESHOLD)
        )
        if (
            features["parallel_definition_score"] > 0
            and similarity >= PARALLEL_DEFINITION_RELATED_THRESHOLD
        ):
            confidences["related"] = max(confidences["related"], EDGE_CONFIDENCE_THRESHOLD)

    strongest_edge_confidence = max(
        confidences["branch"],
        confidences["continuation"],
        confidences["related"],
    )
    confidences["unrelated"] = round(max(0.0, 1.0 - strongest_edge_confidence), 3)

    return {label: round(confidence, 3) for label, confidence in confidences.items()}


def select_edge_label(confidences: dict) -> tuple[str, float] | None:
    edge_confidences = {
        label: confidence
        for label, confidence in confidences.items()
        if label != "unrelated"
    }
    best_label = max(edge_confidences, key=edge_confidences.get)
    best_confidence = edge_confidences[best_label]

    if confidences["unrelated"] >= best_confidence:
        return None
    if best_confidence < EDGE_CONFIDENCE_THRESHOLD:
        return None

    return best_label, best_confidence


def analyze_immediate_relationship(
    previous_user_text: str,
    previous_ai_text: str,
    user_text: str,
    similarity: float | None = None,
) -> dict:
    if similarity is None:
        previous_embedding = model.encode(previous_user_text + " " + previous_ai_text)
        user_embedding = model.encode(user_text)
        similarity = model.similarity(
            user_embedding.reshape(1, -1),
            previous_embedding.reshape(1, -1),
        ).item()

    features = extract_discourse_features(user_text, previous_user_text, previous_ai_text)
    label_scores = score_edge_labels(similarity, features)
    cross_scores = score_cross_encoder_labels(previous_user_text, previous_ai_text, user_text)
    confidences = judge_edge_label(similarity, features, label_scores, cross_scores)
    selected_edge = select_edge_label(confidences) if allows_candidate(similarity, features) else None

    return {
        "candidate_allowed": allows_candidate(similarity, features),
        "similarity": round(similarity, 3),
        "features": features,
        "heuristic_scores": {label: round(score, 3) for label, score in label_scores.items()},
        "cross_encoder_scores": cross_scores,
        "confidences": confidences,
        "selected_label": selected_edge[0] if selected_edge is not None else "unrelated",
        "selected_confidence": selected_edge[1] if selected_edge is not None else confidences["unrelated"],
    }


def classify_semantic_cross_link(parent_id: int, user_text: str, similarity: float) -> tuple[str, float]:
    parent = turns[parent_id]
    features = extract_discourse_features(user_text, parent.user_text, parent.ai_text)
    cross_scores = score_cross_encoder_labels(parent.user_text, parent.ai_text, user_text)
    if has_comparison_with_prior_topic(features):
        label_scores = score_edge_labels(similarity, features)
        confidence = (
            0.6 * score_to_confidence(label_scores["continuation"], CONTINUATION_MIN_SCORE)
            + 0.4 * cross_scores["continuation"]
        )
        return "continuation", round(max(confidence, EDGE_CONFIDENCE_THRESHOLD), 3)
    return "related", round(max(similarity, EDGE_CONFIDENCE_THRESHOLD), 3)


def immediatePreviousTurnClassifier(
    embedding, 
    timeline_parent_id: int, 
    user_text: str
) -> tuple[int, str, float] | None:
    """
    3-layer classifier for the immediate previous turn.
    
    Layer 1: Embedding similarity (candidate gating)
    Layer 2: Discourse feature scoring (heuristic baseline)
    Layer 3: Cross-encoder final relationship confidence
    
    Returns (parent_id, "continuation" | "branch" | "related", confidence) or None.
    """
    parent = turns[timeline_parent_id]
    
    # Layer 1: Embedding similarity (candidate filter)
    score_to_parent = model.similarity(embedding.reshape(1, -1), parent.embedding.reshape(1, -1)).item()
    
    features = extract_discourse_features(user_text, parent.user_text, parent.ai_text)

    if not allows_candidate(score_to_parent, features):
        # Too dissimilar — no connection, new root
        return None
    
    # Layer 2: Discourse feature scoring (heuristic baseline)
    label_scores = score_edge_labels(score_to_parent, features)
    
    # Layer 3: cross-encoder label-specific confidence
    cross_scores = score_cross_encoder_labels(parent.user_text, parent.ai_text, user_text)
    confidences = judge_edge_label(score_to_parent, features, label_scores, cross_scores)
    judged_label = select_edge_label(confidences)
    if judged_label is not None:
        label, confidence = judged_label
        return (timeline_parent_id, label, confidence)
    
    return None


def find_semantic_parents(embedding, timeline_parent: int | None, user_text: str) -> list[tuple[int, str, float]]:
    """
    Find all turns the new turn should attach to in the graph.

    Step 1: classify against the immediately preceding turn using discourse analysis.
    Step 2: scan all previous turns for other high-similarity matches (cross-thread related).

    Returns a list of (turn_id, "continuation" | "branch" | "related", confidence) edges.
    """
    if not turns:
        return []

    # step 1: relationship with the immediately previous turn (2-layer classifier)
    timeline_result = immediatePreviousTurnClassifier(embedding, timeline_parent, user_text) if timeline_parent is not None else None

    # step 2: compare against all previous turns at once
    all_embeddings = np.array([t.embedding for t in turns])
    scores = model.similarity(embedding.reshape(1, -1), all_embeddings)[0].tolist()

    # rank all turns from most to least similar
    scored = sorted(zip([t.id for t in turns], scores), key=lambda x: x[1], reverse=True)

    best_score = scored[0][1]
    semantic_results = []
    has_comparison_question = any(pattern in user_text.lower() for pattern in COMPARISON_PATTERNS)

    if best_score >= THRESHOLD or has_comparison_question:
        for id, score in scored:
            if len(semantic_results) >= MAX_PARENTS:
                break
            if id == timeline_parent:
                # already handled in step 1
                continue
            cross_features = extract_discourse_features(user_text, turns[id].user_text, turns[id].ai_text)
            if has_comparison_with_prior_topic(cross_features):
                label, confidence = classify_semantic_cross_link(id, user_text, score)
                semantic_results.append((id, label, confidence))
            elif score >= best_score - CLOSE_GAP:
                # semantically close but not the timeline parent → classify cross-link
                label, confidence = classify_semantic_cross_link(id, user_text, score)
                semantic_results.append((id, label, confidence))

    # timeline result goes first, related cross-links after
    if timeline_result is not None:
        return [timeline_result] + semantic_results
    return semantic_results


def add_turn(user_text: str, ai_text: str) -> Turn:
    """
    Create a new turn from a user message and AI response, attach it to the graph,
    and assign it to the appropriate concept(s).
    """
    global concept_counter

    # the previous turn in the timeline (always the last one added)
    timeline_parent = turns[-1].id if turns else None

    # use the new user question to classify its relationship to existing turns
    user_embedding = model.encode(user_text)

    # store the full turn embedding for future semantic context
    combined = user_text + " " + ai_text
    embedding = model.encode(combined)

    # find which previous turns this turn relates to, and how
    semantic_parents = find_semantic_parents(user_embedding, timeline_parent, user_text)

    # no semantic parents → this turn starts a new topic
    root = len(semantic_parents) == 0

    if root:
        # new concept: assign the next available concept id
        concept_ids = [concept_counter]
        concept_counter += 1
    else:
        # inherit concept ids from all semantic parents (deduplicated)
        # if parents span two concepts, this turn belongs to both
        seen = set()
        concept_ids = []
        for parent_id, _, _ in semantic_parents:
            for cid in turns[parent_id].concept_ids:
                if cid not in seen:
                    seen.add(cid)
                    concept_ids.append(cid)

    turn = Turn(
        id=next_id(),
        root=root,
        embedding=embedding,
        user_text=user_text,
        ai_text=ai_text,
        timestamp=datetime.now(),
        timeline_parent=timeline_parent,
        semantic_parents=semantic_parents,
        concept_ids=concept_ids,
    )
    turns.append(turn)
    return turn


def reset_conversation():
    global concept_counter

    turns.clear()
    concept_counter = 0


def reclassify_turns() -> list[Turn]:
    existing_turns = [
        (turn.user_text, turn.ai_text, turn.timestamp)
        for turn in turns
    ]

    reset_conversation()

    rebuilt_turns = []
    for user_text, ai_text, timestamp in existing_turns:
        turn = add_turn(user_text, ai_text)
        turn.timestamp = timestamp
        rebuilt_turns.append(turn)

    return rebuilt_turns


def main():
    print("Conversation tree (type 'quit' to exit, 'history' to see turns)\n")
    while True:
        user_input = input("u: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "history":
            for t in turns:
                print(f"  {t}")
            continue

        ai_input = input("a: ").strip()
        if not ai_input:
            continue

        add_turn(user_input, ai_input) 


if __name__ == "__main__":
    main()
