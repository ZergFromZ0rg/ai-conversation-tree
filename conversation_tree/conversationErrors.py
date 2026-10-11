"""The error contract between the conversation services and whatever hosts them.

Services raise these instead of returning ``None`` / ``False`` for a missing
thing or leaking transport exceptions, so a caller (the HTTP layer, a CLI,
another application) can tell the three failure kinds apart without knowing
how the services are implemented. Each also subclasses the closest builtin, so
code that already catches ``ValueError`` / ``RuntimeError`` keeps working.
"""


class ConversationTreeError(Exception):
    """Base class: catch this to handle any expected service failure."""


class NotFoundError(ConversationTreeError, LookupError):
    """The conversation, edge, concept, or link the caller named doesn't exist."""


class InvalidInputError(ConversationTreeError, ValueError):
    """The request is well-formed but breaks a domain rule (bad label, bad range)."""


class ProviderError(ConversationTreeError, RuntimeError):
    """The model provider failed, is unreachable, or is not configured."""
