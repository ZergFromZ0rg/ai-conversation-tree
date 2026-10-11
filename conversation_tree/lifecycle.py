"""Starting the package up: configure it, create the schema, optionally warm the models."""

from .config import Settings, configure


def init(settings: Settings | None = None) -> Settings:
    """Configure the package and make sure its database exists.

    Idempotent, and safe to call again with different settings (it drops every
    cache that belonged to the previous ones). With no argument it reads the
    environment, as the bundled app does. Does not load the embedding models;
    see ``warmup``.
    """
    from . import db, graphStore, vectorStore

    settings = settings or Settings.fromEnv()
    configure(settings)
    graphStore.clear()
    vectorStore.reset()
    db.initDb()
    return settings


def warmup() -> None:
    """Load the embedding models now, so the first classification isn't slow."""
    from .graphService import loadModels

    loadModels()
