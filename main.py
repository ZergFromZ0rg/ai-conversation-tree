from datetime import datetime
from tree import Turn

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
turns: list[Turn] = []

concept_counter = 0

THRESHOLD = 0.6
CLOSE_GAP = 0.05
WEAK_THRESHOLD = 0.35
MAX_PARENTS = 2
SHORT_MSG_WORDS = 5


def next_id():
    return len(turns)


def classify_vs_timeline(embedding, timeline_parent_id: int, user_text: str) -> tuple[int, str] | None:
    """
    Decide how the new turn relates to the immediately preceding turn.

    Returns (parent_id, "continuation") if the new turn stays on the same topic,
            (parent_id, "branch")       if it picks up a sub-topic from the parent's answer,
            None                        if it's too dissimilar to attach (new root).
    """
    parent = turns[timeline_parent_id]

    # how similar is the new turn to the previous one?
    score_to_parent = model.similarity(embedding.reshape(1, -1), parent.embedding.reshape(1, -1)).item()

    # short user messages (e.g. "explain more", "why?") get a relaxed threshold
    # so they don't accidentally become new roots just because they're brief
    is_short = len(user_text.split()) <= SHORT_MSG_WORDS
    effective_threshold = WEAK_THRESHOLD / 2 if is_short else WEAK_THRESHOLD

    if score_to_parent < effective_threshold:
        # too dissimilar — no attachment, this turn starts a new concept
        return None

    grandparent_id = parent.timeline_parent
    if grandparent_id is None:
        # only one prior turn exists, must be a continuation
        return (timeline_parent_id, "continuation")

    grandparent = turns[grandparent_id]

    # how similar is the new turn to the turn before the parent?
    score_to_grandparent = model.similarity(embedding.reshape(1, -1), grandparent.embedding.reshape(1, -1)).item()

    # if the new turn is close to the parent but NOT to the grandparent,
    # it means it's picking up a sub-topic introduced in the parent's answer → branch
    # example: (what is DTMC? / DTMC uses T matrix) → (what is T matrix? / ...) = branch
    if score_to_parent > THRESHOLD and score_to_grandparent < THRESHOLD:
        return (timeline_parent_id, "branch")

    # otherwise it's staying on the same topic → continuation
    return (timeline_parent_id, "continuation")


def find_semantic_parents(embedding, timeline_parent: int | None, user_text: str) -> list[tuple[int, str]]:
    """
    Find all turns the new turn should attach to in the graph.

    Step 1: classify against the immediately preceding turn (timeline parent).
    Step 2: scan all previous turns for any other high-similarity matches
            that aren't the timeline parent — these are cross-thread branches.

    Returns a list of (turn_id, "continuation" | "branch") pairs.
    """
    if not turns:
        return []

    # step 1: relationship with the immediately previous turn
    timeline_result = classify_vs_timeline(embedding, timeline_parent, user_text) if timeline_parent is not None else None

    # step 2: compare against all previous turns at once
    all_embeddings = np.array([t.embedding for t in turns])
    scores = model.similarity(embedding.reshape(1, -1), all_embeddings)[0].tolist()

    # rank all turns from most to least similar
    scored = sorted(zip([t.id for t in turns], scores), key=lambda x: x[1], reverse=True)

    best_score = scored[0][1]
    semantic_results = []

    if best_score >= THRESHOLD:
        for id, score in scored:
            if len(semantic_results) >= MAX_PARENTS:
                break
            if id == timeline_parent:
                # already handled in step 1
                continue
            if score >= best_score - CLOSE_GAP:
                # semantically close but not the timeline parent → cross-thread branch
                semantic_results.append((id, "branch"))

    # timeline result goes first, cross-thread branches after
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

    # embed the full turn (user + AI) for richer semantic context
    # this means branching detection captures sub-topics the AI introduced
    combined = user_text + " " + ai_text
    embedding = model.encode(combined)

    # find which previous turns this turn relates to, and how
    semantic_parents = find_semantic_parents(embedding, timeline_parent, user_text)

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
        for parent_id, _ in semantic_parents:
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