import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from main import addTurn, resetConversation, turns


def olderEdgeSet(turn):
    return {
        (parentId, label)
        for parentId, label, _ in turn.semanticParents
        if parentId != turn.timelineParent
    }


def runCase(name, conversation, expectedEdges):
    resetConversation()
    created = []
    for userText, aiText in conversation:
        created.append(addTurn(userText, aiText))

    targetTurn = created[-1]
    actualEdges = olderEdgeSet(targetTurn)
    matches = actualEdges == expectedEdges
    status = "PASS" if matches else "FAIL"
    print(
        f"{status:4}  {name}: expected={sorted(expectedEdges)} "
        f"predicted={sorted(actualEdges)}"
    )
    return matches


def main():
    cases = [
        (
            "older related concept retrieval",
            [
                ("What is Python?", "Python is a programming language."),
                ("How do I define a function in Python?", "Use the def keyword."),
                ("What is CSS?", "CSS styles web pages."),
                ("How do I center a div?", "Use flexbox or grid."),
                ("What are Python classes?", "Classes define object blueprints."),
            ],
            {(0, "related")},
        ),
        (
            "older continuation comparison retrieval",
            [
                ("What is TCP?", "TCP is a reliable transport protocol."),
                ("What is UDP?", "UDP is a lightweight transport protocol."),
                ("How do I sort a list in Python?", "Use list.sort() or sorted()."),
                ("What is an index in SQL?", "An index speeds up queries."),
                ("How does TCP compare to UDP?", "TCP is reliable; UDP is lower latency."),
            ],
            {(0, "continuation"), (1, "continuation")},
        ),
        (
            "no unrelated older cross link",
            [
                ("How do I center a div?", "Use flexbox."),
                ("What is TCP?", "TCP is a transport protocol."),
                ("What are cats?", "Cats are domesticated mammals."),
                ("How do I bake bread?", "Mix, knead, proof, and bake."),
            ],
            set(),
        ),
    ]

    correct = 0
    for name, conversation, expectedEdges in cases:
        correct += int(runCase(name, conversation, expectedEdges))

    print(f"\nAccuracy: {correct}/{len(cases)} = {correct / len(cases):.1%}")


if __name__ == "__main__":
    main()
