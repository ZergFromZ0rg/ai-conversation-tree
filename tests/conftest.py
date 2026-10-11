"""Shared fixtures. The environment is set before any app module is imported:
db.py reads AI_CONVERSATION_TREE_DB at import time, and the models must load
from the local Hugging Face cache (no network in tests)."""

import os
import tempfile
from pathlib import Path

_dataDir = tempfile.mkdtemp(prefix="act-tests-")
os.environ["AI_CONVERSATION_TREE_DB"] = str(Path(_dataDir) / "test.db")
os.environ["AI_CONVERSATION_TREE_STUB_LLM"] = "1"
os.environ.pop("AI_CONVERSATION_TREE_API_TOKEN", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import asyncio

import httpx

from conversation_tree import api
from conversation_tree.db import listConversations
from conversation_tree.chatService import deleteConversationSession


@pytest.fixture(scope="session")
def _app():
    return api.app


class SyncClient:
    """Minimal sync wrapper over httpx's ASGI transport. Starlette's own
    TestClient (0.35, pinned by fastapi 0.109) breaks on httpx >= 0.28."""

    def __init__(self, app, headers=None):
        self._app = app
        self._headers = headers or {}

    def request(self, method, url, **kwargs):
        async def run():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver", headers=self._headers
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(run())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def patch(self, url, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)


@pytest.fixture()
def client(_app):
    """A client against a clean database: every conversation is removed after
    the test so tests don't see each other's data."""
    yield SyncClient(_app)
    for conversation in listConversations():
        deleteConversationSession(conversation["id"])


def chat(*pairs):
    """[(user, assistant), ...] -> the role-tagged message list the ingest API takes."""
    messages = []
    for user, assistant in pairs:
        messages.append({"role": "user", "text": user})
        messages.append({"role": "assistant", "text": assistant})
    return messages
