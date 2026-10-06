"""Load a small demo workspace through the public ingestion API.

    python scripts/seed_demo.py [--server http://127.0.0.1:8000] [--token ...]

It posts four chats exactly as the browser extension would (one per source),
so running it also exercises ingestion end to end. Two pairs of chats overlap
on purpose (Postgres indexing, Python decorators) so the "also discussed
elsewhere" panel and the workspace map have links to draw, and the migration
chat contains a repeated question and an "as I said" so context-health has
something to flag. Re-running is safe: each chat is matched by its external id.

Start the server with a small context window to see the horizon marker:

    AI_CONVERSATION_TREE_CONTEXT_TOKENS=400 uvicorn api:app
"""

import argparse
import sys

import httpx

MIGRATION = {
    "source": "chatgpt",
    "externalId": "demo-pg-migration",
    "title": "Planning a Postgres migration",
    "url": "https://chatgpt.com/c/demo-pg-migration",
    "turns": [
        (
            "We need to migrate a 2 TB MySQL database to Postgres with zero downtime. Where do I start?",
            "Start with a compatibility audit: list MySQL-specific types, stored procedures, and SQL dialect quirks. Then stand up Postgres, replicate changes with a CDC tool such as Debezium or pglogical, validate row counts and checksums continuously, and rehearse the cutover on a staging copy. Keep the old database writable until the replica has been in sync for several days.",
        ),
        (
            "Which tool should I use to keep the two databases in sync during the move?",
            "For MySQL to Postgres, Debezium reading the binlog and writing through Kafka Connect is the most common route. It captures every row change in order, supports schema evolution, and lets you replay from an offset if the sink falls behind. pgloader is simpler for the initial bulk copy but is not a continuous replication tool.",
        ),
        (
            "How should I think about indexes on the Postgres side?",
            "Do not copy MySQL indexes blindly. Postgres supports partial indexes, expression indexes, and several index types: B-tree for equality and range, GIN for arrays and full text, BRIN for huge append-only tables. Build indexes after the bulk load to keep the copy fast, and use CREATE INDEX CONCURRENTLY on live tables so writes are never blocked.",
        ),
        (
            "What is the safest way to cut over the application?",
            "Use a read-only window as short as you can make it: pause writers, wait for replication lag to hit zero, verify checksums on the hottest tables, flip the connection string, and resume. Keep reverse replication running for a week so you can roll back without data loss.",
        ),
        (
            "As I said, we cannot have any downtime at all, so a pause-writers window is out. What else is there?",
            "Then use dual writes behind a feature flag: the application writes to both databases, reads shift gradually from MySQL to Postgres by percentage, and a reconciliation job compares the two continuously. It is more engineering work, but there is no moment when writes stop.",
        ),
        (
            "How should I think about indexes on the Postgres side?",
            "Do not copy MySQL indexes blindly. Postgres supports partial indexes, expression indexes, and several index types: B-tree for equality and range, GIN for arrays and full text, BRIN for huge append-only tables. Build indexes after the bulk load to keep the copy fast, and use CREATE INDEX CONCURRENTLY on live tables so writes are never blocked.",
        ),
        (
            "What should the rollback plan look like for the dual-write approach?",
            "Rollback is flipping the read percentage back to zero and turning the Postgres write path off, because MySQL was never stopped receiving writes. Keep the reconciliation job running afterwards to prove nothing drifted, and only decommission MySQL after a full billing cycle without discrepancies.",
        ),
    ],
}

DECORATORS = {
    "source": "claude",
    "externalId": "demo-decorators",
    "title": "Python decorators",
    "url": "https://claude.ai/chat/demo-decorators",
    "turns": [
        (
            "How do decorators work in Python?",
            "A decorator is a function that takes another function and returns a replacement for it. The @name syntax above a def is shorthand for func = name(func). Typical uses are logging, caching, timing, and access checks.",
        ),
        (
            "Can you show an example that times a function?",
            "Define timed(fn) with an inner wrapper(*args, **kwargs) that records time.perf_counter() before and after calling fn, prints the elapsed time, and returns fn's result. Apply it with @timed. Add functools.wraps(fn) to the wrapper so the name and docstring survive.",
        ),
        (
            "Why does functools.wraps matter there?",
            "Without wraps, the decorated function reports the wrapper's name and loses its docstring and signature, which breaks help(), debuggers, and frameworks that introspect functions. wraps copies those attributes onto the wrapper.",
        ),
    ],
}

INDEXING = {
    "source": "gemini",
    "externalId": "demo-db-indexing",
    "title": "Database indexing deep dive",
    "url": "https://gemini.google.com/app/demo-db-indexing",
    "turns": [
        (
            "Explain how B-tree indexes work in PostgreSQL.",
            "A B-tree index keeps keys sorted in a balanced tree of pages. A lookup walks from the root to a leaf in logarithmic time, and leaves are linked so range scans read neighbouring pages in order. Postgres uses B-tree for equality and range queries by default.",
        ),
        (
            "When would I choose a GIN index instead?",
            "Choose GIN when one row holds many searchable values: array columns, jsonb documents, and full-text search vectors. A GIN index maps each value to the list of rows containing it, which makes containment queries fast at the cost of slower writes.",
        ),
        (
            "What does CREATE INDEX CONCURRENTLY do?",
            "It builds the index without taking a lock that blocks writes, by scanning the table twice and waiting out in-flight transactions. It takes longer and cannot run inside a transaction block, but it is the right choice for a live production table.",
        ),
        (
            "How should I think about indexes on the Postgres side when migrating from MySQL?",
            "Do not copy MySQL indexes blindly. Rebuild them for Postgres after the bulk load, consider partial and expression indexes where MySQL needed workarounds, and use CREATE INDEX CONCURRENTLY on live tables so writes are never blocked.",
        ),
    ],
}

DECORATOR_PATTERNS = {
    "source": "claude",
    "externalId": "demo-decorator-patterns",
    "title": "Caching with decorators",
    "url": "https://claude.ai/chat/demo-decorator-patterns",
    "turns": [
        (
            "How do I cache function results with a Python decorator?",
            "Use functools.lru_cache for pure functions: @lru_cache(maxsize=256) memoises results keyed by the arguments. For a custom cache, write a decorator that keeps a dict in its closure and checks it before calling the wrapped function.",
        ),
        (
            "What are the pitfalls of caching with decorators in Python?",
            "Arguments must be hashable, cached values keep objects alive, and methods cache per instance unless you handle self carefully. Never cache functions with side effects or time-dependent results without an expiry.",
        ),
    ],
}

CHATS = [MIGRATION, DECORATORS, INDEXING, DECORATOR_PATTERNS]


def messages(turns):
    flat = []
    for user, assistant in turns:
        flat.append({"role": "user", "text": user})
        flat.append({"role": "assistant", "text": assistant})
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="", help="API token, if the server requires one")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    server = args.server.rstrip("/")
    with httpx.Client(timeout=300, headers=headers) as client:
        for chat in CHATS:
            body = {
                "source": chat["source"],
                "externalId": chat["externalId"],
                "title": chat["title"],
                "url": chat["url"],
                "messages": messages(chat["turns"]),
            }
            try:
                response = client.post(f"{server}/ingest/conversations", json=body)
            except httpx.HTTPError as error:
                print(f"Couldn't reach {server}: {error}", file=sys.stderr)
                return 1
            if response.status_code != 200:
                print(f"{chat['title']}: HTTP {response.status_code} {response.text}", file=sys.stderr)
                return 1
            result = response.json()
            print(
                f"{chat['source']:8} {chat['title']:34} conversation {result['conversationId']}: "
                f"+{result['addedTurns']} turns ({result['turnCount']} total)"
            )
        print(f"\nOpen {server}/ui")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
