"""Conversation graph logic: classify turns, link concepts, analyse context.

Importing the package is cheap: no environment is read, no database is opened,
and no model is loaded. A host calls ``init(Settings(...))`` once, and
``warmup()`` if it wants the embedding models loaded up front instead of on the
first classification.
"""

from .config import Settings
from .lifecycle import init, warmup

__version__ = "0.1.0"
__all__ = ["Settings", "init", "warmup", "__version__"]
