"""Configuration and start-up: nothing happens at import, everything is explicit."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

import conversation_tree
from conversation_tree import Settings, api, config, db, graphService, graphStore, vectorStore
from conversation_tree.conversationErrors import ConversationTreeError, NotConfiguredError
from conversation_tree.contextService import buildContextHealth
from conversation_tree.ingestService import ingestConversation
from conftest import chat

REPO = Path(__file__).resolve().parent.parent


# --- Settings -------------------------------------------------------------

def test_explicit_settings_are_minimal_and_keep_pgvector_off(tmp_path):
    settings = Settings(dbPath=str(tmp_path / "x.db"))
    assert settings.dbPath == tmp_path / "x.db"  # coerced to a Path
    assert settings.contextWindowTokens == 8000
    assert settings.pgvectorUrl is None


def test_settings_from_env_reads_each_documented_variable():
    settings = Settings.fromEnv(
        {
            "AI_CONVERSATION_TREE_DB": "/data/tree.db",
            "AI_CONVERSATION_TREE_CONTEXT_TOKENS": "400",
            "PGVECTOR_DATABASE_URL": "postgresql://pg/one",
            "DATABASE_URL": "postgresql://pg/two",
        }
    )
    assert settings == Settings(
        dbPath=Path("/data/tree.db"), contextWindowTokens=400, pgvectorUrl="postgresql://pg/one"
    )
    assert Settings.fromEnv({"DATABASE_URL": "postgresql://pg/two"}).pgvectorUrl == "postgresql://pg/two"


def test_settings_from_empty_env_use_the_documented_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = Settings.fromEnv({})
    assert settings.dbPath == tmp_path / ".data" / "conversationTree.db"
    assert settings.contextWindowTokens == 8000
    assert settings.pgvectorUrl == config.LOCAL_PGVECTOR_URL


def test_server_options_from_env():
    options = api.ServerOptions.fromEnv(
        {"AI_CONVERSATION_TREE_API_TOKEN": "t", "AI_CONVERSATION_TREE_CORS_ORIGINS": " https://a.example , ,https://b.example"}
    )
    assert options == api.ServerOptions(apiToken="t", corsOrigins=("https://a.example", "https://b.example"))
    assert api.ServerOptions.fromEnv({}) == api.ServerOptions()


# --- unconfigured use fails loudly ----------------------------------------

def test_services_refuse_to_run_before_init(monkeypatch):
    monkeypatch.setattr(config, "_settings", None)
    with pytest.raises(NotConfiguredError, match="init"):
        db.getConnection()
    assert issubclass(NotConfiguredError, ConversationTreeError)
    # pgvector degrades to "off" instead of raising: it is optional.
    assert vectorStore.enabled() is False


# --- init -----------------------------------------------------------------

def test_init_creates_the_database_and_applies_the_settings(reconfigurable, tmp_path):
    target = tmp_path / "nested" / "dir" / "fresh.db"
    returned = conversation_tree.init(Settings(dbPath=target, contextWindowTokens=123))
    assert returned.dbPath == target and target.exists()
    assert config.getSettings().contextWindowTokens == 123
    assert db.listConversations() == []


def test_init_drops_state_that_belonged_to_the_previous_settings(reconfigurable, tmp_path):
    conversationId = ingestConversation("other", "cache", chat(("What are cats?", "Small mammals.")))["conversationId"]
    with graphStore.lockedGraph(conversationId):
        pass  # leaves the loaded graph in the cache
    assert graphStore._cache
    vectorStore._schemaReady = True
    conversation_tree.init(Settings(dbPath=tmp_path / "other.db"))
    assert graphStore._cache == {}
    assert vectorStore._schemaReady is False


def test_the_context_window_comes_from_settings(reconfigurable, tmp_path):
    conversation_tree.init(Settings(dbPath=tmp_path / "ctx.db", contextWindowTokens=555))
    conversationId = ingestConversation("other", "ctx", chat(("What are cats?", "Small mammals.")))["conversationId"]
    assert buildContextHealth(conversationId)["windowTokens"] == 555
    assert buildContextHealth(conversationId, windowTokens=77)["windowTokens"] == 77


def test_two_databases_do_not_see_each_other(reconfigurable, tmp_path):
    conversation_tree.init(Settings(dbPath=tmp_path / "a.db"))
    ingestConversation("other", "only-in-a", chat(("What are cats?", "Small mammals.")))
    assert len(db.listConversations()) == 1
    conversation_tree.init(Settings(dbPath=tmp_path / "b.db"))
    assert db.listConversations() == []


# --- lazy models ----------------------------------------------------------

def test_models_load_once_and_are_shared():
    assert graphService.loadModels() is graphService.loadModels()
    assert graphService.biEncoder() is graphService.loadModels()[0]
    assert graphService.crossEncoder() is graphService.loadModels()[1]


def test_app_lifespan_initialises_and_warms_up(reconfigurable, tmp_path):
    target = tmp_path / "lifespan.db"
    app = api.createApp(settings=Settings(dbPath=target), options=api.ServerOptions())

    async def run():
        async with app.router.lifespan_context(app):
            return config.getSettings().dbPath

    assert asyncio.run(run()) == target
    assert target.exists()
    assert graphService._loadedModels is not None


# --- importing does nothing -----------------------------------------------

def test_importing_the_package_reads_no_config_opens_no_database_and_loads_no_model(tmp_path):
    marker = tmp_path / "must-not-exist.db"
    script = (
        "import sys\n"
        "import conversation_tree\n"
        "from conversation_tree import api, chatService, graphStore, analysisService, ingestService, contextService\n"
        "from conversation_tree import graphService\n"
        "assert graphService._loadedModels is None, 'models loaded at import'\n"
        "heavy = [name for name in ('sentence_transformers', 'torch') if name in sys.modules]\n"
        "assert not heavy, f'imported at import time: {heavy}'\n"
    )
    environment = {
        **os.environ,
        "AI_CONVERSATION_TREE_DB": str(marker),
        "AI_CONVERSATION_TREE_API_TOKEN": "would-be-read-at-import",
    }
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, env=environment, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
