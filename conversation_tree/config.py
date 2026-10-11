"""Runtime configuration, supplied once by the host through ``init()``.

Nothing in the package reads the environment or opens a database while it is
being imported. A host builds a ``Settings`` (explicitly, or from the
environment with ``Settings.fromEnv()``) and passes it to
``conversation_tree.init``. Services that need a setting ask for it here, so
two hosts can run with different databases without sharing any module state.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from .conversationErrors import NotConfiguredError

DEFAULT_CONTEXT_WINDOW_TOKENS = 8000
# The local Compose database in docker-compose.pgvector.yml.
LOCAL_PGVECTOR_URL = "postgresql://conversation_tree:conversation_tree@127.0.0.1:5432/conversation_tree"


@dataclass(frozen=True)
class Settings:
    dbPath: Path
    """The SQLite file. Its directory is created on init."""

    contextWindowTokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    """Assumed size of the span a model attends to well. Context-loss detection
    measures against it, and it caps how much earlier conversation is sent with
    each new message."""

    pgvectorUrl: str | None = None
    """PostgreSQL URL for pgvector retrieval. ``None`` keeps retrieval local (NumPy)."""

    def __post_init__(self):
        object.__setattr__(self, "dbPath", Path(self.dbPath))

    @classmethod
    def fromEnv(cls, environ=None) -> "Settings":
        """The settings the bundled app and CLI use.

        - ``AI_CONVERSATION_TREE_DB``: database file (default ``.data/conversationTree.db``
          under the working directory)
        - ``AI_CONVERSATION_TREE_CONTEXT_TOKENS``: context window (default 8000)
        - ``PGVECTOR_DATABASE_URL`` / ``DATABASE_URL``: pgvector database (default: the
          local Compose database; retrieval falls back to NumPy if it is unreachable)
        """
        environ = os.environ if environ is None else environ
        return cls(
            dbPath=environ.get("AI_CONVERSATION_TREE_DB") or Path.cwd() / ".data" / "conversationTree.db",
            contextWindowTokens=int(
                environ.get("AI_CONVERSATION_TREE_CONTEXT_TOKENS") or DEFAULT_CONTEXT_WINDOW_TOKENS
            ),
            pgvectorUrl=environ.get("PGVECTOR_DATABASE_URL")
            or environ.get("DATABASE_URL")
            or LOCAL_PGVECTOR_URL,
        )


_settings: Settings | None = None


def configure(settings: Settings) -> None:
    global _settings
    _settings = settings


def currentSettings() -> Settings | None:
    """The active settings, or None before ``init()``. For code that can degrade gracefully."""
    return _settings


def getSettings() -> Settings:
    if _settings is None:
        raise NotConfiguredError(
            "conversation_tree is not configured: call conversation_tree.init(Settings(dbPath=...)) first."
        )
    return _settings
