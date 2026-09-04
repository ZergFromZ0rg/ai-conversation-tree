"""Accuracy harness for cross-conversation concept linking.

Each case builds a handful of small conversations in a throwaway database,
rebuilds every 'auto' concept link, and checks which conversations ended up
linked (by title) and, where it matters, the link kind. Run:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python eval_concept_links.py
"""

import os
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import conceptIndex
import db
from chatService import persistGraphTurn
from graphService import ConversationGraph, addTurn


def buildConversation(title, messages):
    conversationId = db.createConversation(title)
    graph = ConversationGraph()
    for userText, aiText in messages:
        turn = addTurn(graph, userText, aiText)
        persistGraphTurn(conversationId, turn)
    return conversationId


def runCase(name, conversations, expectedPairs, expectedKind=None):
    db.dbPath = Path(tempfile.gettempdir()) / f"eval_concept_links_{uuid.uuid4().hex}.db"
    db.initDb()
    conceptIndex._lastRelink.clear()

    titleById = {}
    for title, messages in conversations:
        titleById[buildConversation(title, messages)] = title

    conceptIndex.relinkAllConceptLinks()
    links = db.listAllConceptLinks()

    actualPairs = {
        frozenset((titleById[link["aConversationId"]], titleById[link["bConversationId"]]))
        for link in links
    }
    kinds = {link["label"] for link in links}

    ok = actualPairs == expectedPairs
    if expectedKind is not None:
        ok = ok and kinds <= {expectedKind}

    status = "PASS" if ok else "FAIL"
    detail = f"expected={_fmt(expectedPairs)} predicted={_fmt(actualPairs)}"
    if expectedKind is not None:
        detail += f" kind={sorted(kinds)}"
    print(f"{status:4}  {name}: {detail}")

    for suffix in ("", "-wal", "-shm"):
        Path(str(db.dbPath) + suffix).unlink(missing_ok=True)
    return ok


def _fmt(pairs):
    return sorted("+".join(sorted(pair)) for pair in pairs)


def main():
    cases = [
        (
            "same topic across chats links",
            [
                (
                    "python-a",
                    [
                        ("What is the Python programming language?", "Python is a high-level language."),
                        ("How do I define a function in Python?", "Use the def keyword."),
                    ],
                ),
                (
                    "python-b",
                    [
                        ("Tell me about Python as a programming language", "Python is popular and readable."),
                        ("What are Python functions and how do they work?", "They group reusable code."),
                    ],
                ),
            ],
            {frozenset(("python-a", "python-b"))},
        ),
        (
            "unrelated topics stay separate",
            [
                ("python", [("What is the Python programming language used for?", "Scripting and data work.")]),
                ("garden", [("When is the best time to prune apple trees?", "Late winter, while dormant.")]),
            ],
            set(),
        ),
        (
            "links are concept-level, not conversation-level",
            [
                (
                    "mixed",
                    [
                        ("What is the Python programming language?", "A high-level language."),
                        ("How do database indexes speed up SQL queries?", "They avoid full scans."),
                    ],
                ),
                ("sql-only", [("Explain how indexes work in a SQL database", "An index is a sorted lookup structure.")]),
                ("py-only", [("What is Python used for as a language?", "Automation, web, and data.")]),
            ],
            {
                frozenset(("mixed", "sql-only")),
                frozenset(("mixed", "py-only")),
            },
        ),
        (
            "trivial concept is not linked",
            [
                ("greeting", [("hi there", "Hello!")]),
                ("python", [("What is the Python programming language used for?", "Scripting and data work.")]),
            ],
            set(),
        ),
        (
            "near-identical questions link as 'same'",
            [
                ("ask-1", [("What are the key characteristics of the Python language?", "Readable, dynamic, batteries-included.")]),
                ("ask-2", [("What are the key characteristics of the Python language?", "Readable, dynamic, batteries-included.")]),
            ],
            {frozenset(("ask-1", "ask-2"))},
            "same",
        ),
    ]

    correct = 0
    for case in cases:
        correct += int(runCase(*case))

    print(f"\nAccuracy: {correct}/{len(cases)} = {correct / len(cases):.1%}")


if __name__ == "__main__":
    main()
