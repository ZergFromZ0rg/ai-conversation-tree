from datetime import datetime
from tree import Message

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
messages: list[Message] = []

concept_counter = 0

# minimum similarity score for a message to be a strong semantic parent
THRESHOLD = 0.6
# how close a runner-up score needs to be to the best score to also qualify as a parent
CLOSE_GAP = 0.05
# minimum similarity to the timeline parent to count as any kind of attachment (below = new root)
WEAK_THRESHOLD = 0.35
MAX_PARENTS = 2
SHORT_MSG_WORDS = 5


def next_id():
    # the next message id is just the current length of the list
    return len(messages)


def classify_vs_timeline(embedding, timeline_parent_id: int, text: str, user) -> tuple[int, str] | None:
    # get the previous message (the one directly before this one in the conversation)
    parent = messages[timeline_parent_id]

    # AI responses always belong to the conversation — skip similarity check
    if not user:
        return (timeline_parent_id, "continuation")

    # measure how similar the new message is to the previous message
    score_to_parent = model.similarity(embedding.reshape(1, -1), parent.embedding.reshape(1, -1)).item()

    is_short = len(text.split()) <= SHORT_MSG_WORDS
    is_opposite_user = user != parent.user

    # short reply from the opposite user (e.g. "fine thanks", "explain more", "???")
    # → use a lower threshold, but still require some similarity so new topics don't slip through
    effective_threshold = WEAK_THRESHOLD / 2 if (is_short and is_opposite_user) else WEAK_THRESHOLD

    if score_to_parent < effective_threshold:
        return None

    # get the message before the previous one (the "question" that generated the "answer")
    grandparent_id = parent.timeline_parent
    if grandparent_id is None:
        # no grandparent means this is only the second message ever → must be a continuation
        return (timeline_parent_id, "continuation")

    grandparent = messages[grandparent_id]

    # measure how similar the new message is to the grandparent (two messages ago)
    score_to_grandparent = model.similarity(embedding.reshape(1, -1), grandparent.embedding.reshape(1, -1)).item()

    # if the new message is close to the answer but NOT to the question that created it,
    # it means the new message is picking up a sub-topic from the answer → branch
    # example: "what is DTMC?" → "DTMC uses a T matrix" → "what is T matrix?" (branch)
    if score_to_parent > THRESHOLD and score_to_grandparent < THRESHOLD:
        return (timeline_parent_id, "branch")

    # otherwise it's staying on the same topic → continuation
    return (timeline_parent_id, "continuation")


def find_semantic_parents(embedding, timeline_parent: int | None, text: str, user) -> list[tuple[int, str]]:
    if not messages:
        return []

    # step 1: decide the relationship with the immediately previous message
    # this returns (parent_id, "continuation" | "branch") or None if unrelated
    timeline_result = classify_vs_timeline(embedding, timeline_parent, text, user) if timeline_parent is not None else None

    # step 2: compare the new message against ALL previous messages at once
    all_embeddings = np.array([msg.embedding for msg in messages])
    scores = model.similarity(embedding.reshape(1, -1), all_embeddings)[0].tolist()

    # sort all (message_id, score) pairs from most to least similar
    scored = sorted(zip([msg.id for msg in messages], scores), key=lambda x: x[1], reverse=True)

    best_score = scored[0][1]
    semantic_results = []

    if best_score >= THRESHOLD:
        for id, score in scored:
            if len(semantic_results) >= MAX_PARENTS:
                break
            if id == timeline_parent:
                # skip — timeline parent is already handled in step 1
                continue
            if score >= best_score - CLOSE_GAP:
                # this message is semantically close but not the timeline parent
                # → it's a cross-thread connection → always a branch
                semantic_results.append((id, "branch"))

    # put the timeline result first, then any cross-thread branches
    if timeline_result is not None:
        return [timeline_result] + semantic_results
    return semantic_results


def add_message(text: str, user: str) -> Message:
    global concept_counter

    # the previous message in the timeline (always the last one added)
    timeline_parent = messages[-1].id if messages else None

    # convert the text into a vector so we can measure similarity
    embedded = model.encode(text)

    # find which previous messages this new message relates to, and how
    semantic_parents = find_semantic_parents(embedded, timeline_parent, text, user)

    # if no semantic parents were found, this message starts a new topic
    root = len(semantic_parents) == 0

    if root:
        # new concept: assign the next available concept id
        concept_id = [concept_counter]
        concept_counter += 1
    else:
        # inherit concept ids from all semantic parents (deduplicated)
        # if parents span two concepts (e.g. 1 + 2), this message belongs to both
        seen = set()
        concept_id = []
        for parent_id, _ in semantic_parents:
            for cid in messages[parent_id].concept_ids:
                if cid not in seen:
                    seen.add(cid)
                    concept_id.append(cid)

    msg = Message(
        id=next_id(),
        root=root,
        embedding=embedded,
        text=text,
        timestamp=datetime.now(),
        user=user,
        timeline_parent=timeline_parent,       # always the previous message
        semantic_parent=semantic_parents,       # list of (id, "continuation"|"branch")
        concept_id=concept_id,                  # list of concept ids this message belongs to
    )
    messages.append(msg)
    return msg


def main():
    x = 0
    print("Conversation tree (type 'quit' to exit, 'history' to see messages)\n")
    while True:
        if x == 0:
            user_input = input("u: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "history":
                for m in messages:
                    print(f"  {m}")
                continue
            add_message(user_input, user=True)
            x = 1
        else:
            user_input = input("a: ").strip()
            if not user_input:
                continue
            add_message(user_input, user=False)
            x = 0


if __name__ == "__main__":
    main()