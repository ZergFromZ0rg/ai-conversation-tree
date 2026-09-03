"""Per-conversation in-memory graph cache with per-conversation locking.

graphService rebuilds a whole ConversationGraph from persisted rows. Doing that
on every turn is O(n) per turn (O(n^2) over a conversation), and the old
module-global state forced one lock across every conversation. This module keeps
a small LRU of loaded graphs and hands out a per-conversation lock so writes to
different conversations run concurrently while writes to the same one serialize.

Assumes a single process (one uvicorn worker). With multiple workers the cache
would diverge; run with --workers 1 or disable the cache.
"""

import threading
from collections import OrderedDict
from contextlib import contextmanager

from graphService import ConversationGraph, loadConversationGraph

maxCachedGraphs = 32

_cache: "OrderedDict[int, ConversationGraph]" = OrderedDict()
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
        graph = _cache.get(conversationId)
        if graph is not None:
            _cache.move_to_end(conversationId)
        return graph


def _storeGraph(conversationId: int, graph: ConversationGraph) -> None:
    with _cacheLock:
        _cache[conversationId] = graph
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
        graph = _cachedGraph(conversationId)
        if graph is None:
            graph = loadConversationGraph(conversationId)
            _storeGraph(conversationId, graph)
        yield graph


def invalidate(conversationId: int) -> None:
    """Drop a conversation's cached graph (after out-of-band edits or deletion)."""
    with _cacheLock:
        _cache.pop(conversationId, None)
