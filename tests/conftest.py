"""Shared fixtures. The package is configured the way a host would configure
it: an explicit Settings passed to init(), against a throwaway database. Only
the model-loading environment is set up front (no network in tests, so the
models must come from the local Hugging Face cache)."""

import os
import tempfile
from pathlib import Path

os.environ["AI_CONVERSATION_TREE_STUB_LLM"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import asyncio

import httpx

import conversation_tree
from conversation_tree import Settings, api, providers
from conversation_tree.db import listConversations
from conversation_tree.chatService import deleteConversationSession


@pytest.fixture(scope="session", autouse=True)
def configured():
    """The package's settings for the whole run (a throwaway database)."""
    return conversation_tree.init(Settings(dbPath=Path(tempfile.mkdtemp(prefix="act-tests-")) / "test.db"))


@pytest.fixture(autouse=True)
def _builtinReplier():
    """Tests may install their own replier; each test starts and ends on the built-in one."""
    providers.setReplier(None)
    yield
    providers.setReplier(None)


@pytest.fixture()
def reconfigurable(configured):
    """For tests that call init() with other settings: restores the session's afterwards."""
    yield
    conversation_tree.init(configured)


@pytest.fixture(scope="session")
def _app(configured):
    # Explicit options, so nothing in the environment can switch auth on.
    return api.createApp(options=api.ServerOptions())


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


@pytest.fixture()
def serverClient():
    """Build a client for an app with specific HTTP options, e.g. serverClient(apiToken="x")."""

    def make(**options):
        return SyncClient(api.createApp(options=api.ServerOptions(**options)))

    return make


def chat(*pairs):
    """[(user, assistant), ...] -> the role-tagged message list the ingest API takes."""
    messages = []
    for user, assistant in pairs:
        messages.append({"role": "user", "text": user})
        messages.append({"role": "assistant", "text": assistant})
    return messages
