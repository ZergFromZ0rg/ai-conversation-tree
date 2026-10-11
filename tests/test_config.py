"""Configuration and start-up: nothing happens at import, everything is explicit."""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import conversation_tree
from conversation_tree import (
    Settings, analysisService, api, conceptIndex, config, db, graphService, graphStore, providers, vectorStore,
)
from conversation_tree.conversationErrors import ConversationTreeError, NotConfiguredError
from conversation_tree.chatService import appendTurn, createConversationSession
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
    conceptIndex._lastRelink[conversationId] = time.monotonic()
    analysisService._summaryCache[(conversationId, 1, 0, "m", 0)] = {"summary": "written for the old database"}

    conversation_tree.init(Settings(dbPath=tmp_path / "other.db"))

    assert graphStore._cache == {}
    assert vectorStore._schemaReady is False
    assert conceptIndex._lastRelink == {}
    assert analysisService._summaryCache == {}


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


def test_a_new_database_is_not_debounced_by_the_old_ones_relink_times(reconfigurable, tmp_path, monkeypatch):
    conversation_tree.init(Settings(dbPath=tmp_path / "first.db"))
    conversationId = createConversationSession()["conversationId"]
    appendTurn(conversationId, "What are cats?", "Small mammals.")
    assert conversationId in conceptIndex._lastRelink, "the turn recorded when this conversation was relinked"

    conversation_tree.init(Settings(dbPath=tmp_path / "second.db"))
    relinked = []
    monkeypatch.setattr(conceptIndex, "relinkConversation", lambda cid: relinked.append(cid) or 0)
    sameId = createConversationSession()["conversationId"]
    assert sameId == conversationId, "a fresh database hands out the same ids again"
    appendTurn(sameId, "What do cats eat?", "Meat.")
    assert relinked == [sameId], "the relink must not be skipped as 'just done' by the other database's timestamp"


# --- settings are validated and made absolute when created ----------------------

def test_a_relative_database_path_is_made_absolute_when_the_settings_are_created(monkeypatch, tmp_path):
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir(); there.mkdir()
    monkeypatch.chdir(here)
    settings = Settings(dbPath="data/../rel/c.db")
    assert settings.dbPath == here / "rel" / "c.db"
    monkeypatch.chdir(there)
    assert settings.dbPath == here / "rel" / "c.db", "it must not follow the working directory afterwards"
    assert Settings(dbPath="~/c.db").dbPath == Path.home() / "c.db"


def test_a_host_that_changes_directory_after_init_keeps_its_database(reconfigurable, monkeypatch, tmp_path):
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir(); there.mkdir()
    monkeypatch.chdir(here)
    conversation_tree.init(Settings(dbPath="rel/c.db"))
    createConversationSession("made before the chdir")
    monkeypatch.chdir(there)
    assert [c["title"] for c in db.listConversations()] == ["made before the chdir"]
    assert not (there / "rel").exists()


@pytest.mark.parametrize("window", [0, -5, 1.5, "8000", True, None])
def test_the_context_window_must_be_a_positive_integer(tmp_path, window):
    with pytest.raises(ValueError, match="contextWindowTokens"):
        Settings(dbPath=tmp_path / "x.db", contextWindowTokens=window)


@pytest.mark.parametrize("value", ["lots", "0", "-3", "1.5"])
def test_a_bad_window_in_the_environment_names_the_variable(value):
    with pytest.raises(ValueError, match="AI_CONVERSATION_TREE_CONTEXT_TOKENS"):
        Settings.fromEnv({"AI_CONVERSATION_TREE_CONTEXT_TOKENS": value})


# --- init is all-or-nothing ---------------------------------------------------

def populate_running_state(conversation_id_source):
    """Put something in every cache init() would otherwise drop."""
    conversationId = conversation_id_source()
    with graphStore.lockedGraph(conversationId):
        pass
    vectorStore._schemaReady = True
    conceptIndex._lastRelink[conversationId] = time.monotonic()
    return conversationId


def test_init_keeps_the_running_configuration_when_the_database_cannot_open(reconfigurable, configured, tmp_path):
    marker = lambda request: "still the old replier"
    providers.setReplier(marker)
    populate_running_state(lambda: ingestConversation("other", "keep", chat(("What are cats?", "Mammals.")))["conversationId"])
    cacheBefore = dict(graphStore._cache)

    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    with pytest.raises(OSError):
        conversation_tree.init(Settings(dbPath=blocker / "inside" / "c.db"), replier=lambda request: "new")

    assert config.getSettings() == configured, "the previous settings are restored"
    assert providers.getReplier()._function is marker, "the previous replier is kept"
    assert dict(graphStore._cache) == cacheBefore and vectorStore._schemaReady is True
    assert conceptIndex._lastRelink, "no cache was dropped for a configuration that never took effect"
    assert db.listConversations(), "and the previous database still answers"


def test_init_with_an_unusable_replier_changes_nothing(reconfigurable, configured, tmp_path):
    target = tmp_path / "never-created.db"
    with pytest.raises(TypeError):
        conversation_tree.init(Settings(dbPath=target), replier=42)
    assert config.getSettings() == configured
    assert not target.exists()


def test_a_failed_first_init_leaves_the_package_unconfigured(reconfigurable, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_settings", None)
    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    with pytest.raises(OSError):
        conversation_tree.init(Settings(dbPath=blocker / "inside" / "c.db"))
    assert config.currentSettings() is None
    with pytest.raises(NotConfiguredError):
        db.getConnection()
