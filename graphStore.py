"""Per-conversation in-memory graph cache with per-conversation locking.

graphService rebuilds a whole ConversationGraph from persisted rows. Doing that
on every turn is O(n) per turn (O(n^2) over a conversation), and the old
module-global state forced one lock across every conversation. This module keeps
a small LRU of loaded graphs and hands out a per-conversation lock so writes to
different conversations run concurrently while writes to the same one serialize.

The cache is process-local, but every entry carries the conversation's SQLite
`updatedAt` version. A worker rechecks that version before using its cached
graph, so writes made by another worker invalidate stale state automatically.
"""

import threading
from collections import OrderedDict
from contextlib import contextmanager

from db import getConversationVersion
from graphService import ConversationGraph, loadConversationGraph

maxCachedGraphs = 32

_cache: "OrderedDict[int, tuple[str, ConversationGraph]]" = OrderedDict()
_cacheLock = threading.Lock()
_conversationLocks: dict[int, threading.RLock] = {}


def _conversationLock(conversationId: int) -> threading.RLock:
    with _cacheLock:
        lock = _conversationLocks.get(conversationId)
        if lock is None:
            lock = threading.RLock()
            _conversationLocks[conversationId] = lock
        return lock


def _cachedGraph(conversationId: int) -> ConversationGraph | None:
    with _cacheLock:
        entry = _cache.get(conversationId)
        if entry is not None:
            _cache.move_to_end(conversationId)
            return entry
        return None


def _storeGraph(conversationId: int, version: str, graph: ConversationGraph) -> None:
    with _cacheLock:
        _cache[conversationId] = (version, graph)
        _cache.move_to_end(conversationId)
        while len(_cache) > maxCachedGraphs:
            _cache.popitem(last=False)


@contextmanager
def lockedGraph(conversationId: int):
    """Hold the conversation's lock and yield its (cached or freshly loaded) graph.

    The graph is left in the cache on exit, so a burst of turns on one
    conversation reloads from the database only once.
    """
    lock = _conversationLock(conversationId)
    with lock:
        version = getConversationVersion(conversationId)
        entry = _cachedGraph(conversationId)
        if entry is None or entry[0] != version:
            graph = loadConversationGraph(conversationId)
            _storeGraph(conversationId, version or "", graph)
        else:
            graph = entry[1]
        yield graph


def invalidate(conversationId: int) -> None:
    """Drop a conversation's cached graph (after out-of-band edits or deletion)."""
    with _cacheLock:
        _cache.pop(conversationId, None)
