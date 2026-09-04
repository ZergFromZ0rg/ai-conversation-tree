import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { useConversation } from "./useConversation";
import { ConversationSidebar } from "./components/ConversationSidebar";
import { ChatTranscript } from "./components/ChatTranscript";
import { Composer } from "./components/Composer";
import { GraphRail } from "./components/GraphRail";
import { GraphDrawer } from "./components/GraphDrawer";
import { WorkspaceMap } from "./components/WorkspaceMap";
import { SettingsModal } from "./components/SettingsModal";

const DRAWER_STORAGE_KEY = "act.drawerOpen";
const SIDEBAR_STORAGE_KEY = "act.sidebarOpen";
const MODEL_STORAGE_KEY = "act.model";
const API_KEYS_STORAGE_KEY = "act.apiKeys";
const NARROW_VIEWPORT_QUERY = "(max-width: 860px)";

// A small curated catalog so the model picker can offer a provider's models
// as soon as the user saves a key for it in Settings, without the backend
// needing to know a browser holds a client-side-only key just to list them.
const CLIENT_MODEL_CATALOG = {
  openai: [
    { id: "openai:gpt-5", label: "OpenAI · gpt-5" },
    { id: "openai:gpt-5-mini", label: "OpenAI · gpt-5-mini" },
  ],
  anthropic: [
    { id: "anthropic:claude-opus-5", label: "Anthropic · Opus 5" },
    { id: "anthropic:claude-sonnet-5", label: "Anthropic · Sonnet 5" },
    { id: "anthropic:claude-haiku-4-5", label: "Anthropic · Haiku 4.5" },
  ],
  gemini: [
    { id: "gemini:gemini-2.5-pro", label: "Gemini · 2.5 Pro" },
    { id: "gemini:gemini-2.5-flash", label: "Gemini · 2.5 Flash" },
  ],
};

function readDrawerPreference() {
  try {
    return window.localStorage.getItem(DRAWER_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function readSidebarPreference() {
  try {
    // Narrow viewports (mobile) always start with the sidebar-as-overlay
    // closed, regardless of what was remembered from a wider session.
    if (window.matchMedia(NARROW_VIEWPORT_QUERY).matches) {
      return false;
    }
    const stored = window.localStorage.getItem(SIDEBAR_STORAGE_KEY);
    return stored === null ? true : stored === "1";
  } catch {
    return true;
  }
}

function readModelPreference() {
  try {
    return window.localStorage.getItem(MODEL_STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

function readApiKeys() {
  try {
    const raw = window.localStorage.getItem(API_KEYS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

export function App() {
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [selectedTurnId, setSelectedTurnId] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(readDrawerPreference);
  const [sidebarOpen, setSidebarOpen] = useState(readSidebarPreference);
  const [mapOpen, setMapOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [apiKeys, setApiKeys] = useState(readApiKeys);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(readModelPreference);

  const {
    turns,
    graph,
    conceptLinks,
    status,
    pendingUserText,
    streamingText,
    sendTurn,
    analyze,
    refresh,
  } = useConversation(conversationId);

  const isBusy = status === "loading" || status === "sending" || status === "analyzing";
  const turnCount = turns.length;
  const threadCount = turns.filter((turn) => turn.root).length;

  const loadConversations = useCallback(async () => {
    const payload = await api.listConversations();
    setConversations(payload.conversations);
    return payload.conversations;
  }, []);

  useEffect(() => {
    void loadConversations().then(async (list) => {
      // An untitled conversation is, by construction, an empty draft (titles
      // are only set once a first turn is sent — see chatService._finalizeTurn).
      // Reuse it instead of spawning a fresh empty draft on every reload.
      const draft = list[0] && !list[0].title ? list[0] : null;
      if (draft) {
        setConversationId((current) => current ?? draft.id);
        return;
      }
      const created = await api.createConversation(null, null);
      setConversationId((current) => current ?? created.conversationId);
      await loadConversations();
    });
  }, [loadConversations]);

  const refreshModels = useCallback(async () => {
    try {
      const payload = await api.listModels();
      setModels(payload.models);
      setSelectedModel((current) => current ?? payload.default ?? null);
    } catch {
      setModels([]);
    }
  }, []);

  useEffect(() => {
    void refreshModels();
  }, [refreshModels]);

  useEffect(() => {
    try {
      if (selectedModel) {
        window.localStorage.setItem(MODEL_STORAGE_KEY, selectedModel);
      }
    } catch {
      /* storage unavailable — ignore */
    }
  }, [selectedModel]);

  const saveApiKeys = useCallback((nextKeys) => {
    setApiKeys(nextKeys);
    try {
      window.localStorage.setItem(API_KEYS_STORAGE_KEY, JSON.stringify(nextKeys));
    } catch {
      /* storage unavailable — ignore */
    }
  }, []);

  // Merge in a provider's curated model catalog once a key is saved for it,
  // so the picker offers those models without the server ever seeing the key.
  const availableModels = [
    ...models,
    ...Object.entries(CLIENT_MODEL_CATALOG).flatMap(([provider, entries]) =>
      apiKeys[provider] ? entries.filter((entry) => !models.some((model) => model.id === entry.id)) : [],
    ),
  ];
  const selectedModelProvider = selectedModel?.split(":")[0];
  const selectedModelApiKey = selectedModelProvider ? apiKeys[selectedModelProvider] : undefined;

  useEffect(() => {
    try {
      window.localStorage.setItem(DRAWER_STORAGE_KEY, drawerOpen ? "1" : "0");
    } catch {
      /* storage unavailable — ignore */
    }
  }, [drawerOpen]);

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_STORAGE_KEY, sidebarOpen ? "1" : "0");
    } catch {
      /* storage unavailable — ignore */
    }
  }, [sidebarOpen]);

  useEffect(() => {
    function onKeyDown(event) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "g") {
        event.preventDefault();
        setDrawerOpen((open) => !open);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const selectConversation = useCallback(
    (id) => {
      // Selecting the conversation already open won't re-trigger useConversation's
      // own [conversationId] effect, so its turns/graph/conceptLinks would go
      // stale after an out-of-band change (e.g. pinning a concept link from the
      // workspace map while already viewing one side of it). Force it here.
      const alreadyOpen = id === conversationId;
      setConversationId(id);
      setSelectedTurnId(null);
      // On a narrow (mobile) viewport the sidebar is a full overlay, so
      // picking a conversation should dismiss it; on a wide viewport it's a
      // docked, independently-toggled panel that a selection shouldn't touch.
      if (window.matchMedia(NARROW_VIEWPORT_QUERY).matches) {
        setSidebarOpen(false);
      }
      const target = conversations.find((conversation) => conversation.id === id);
      if (target?.model) {
        setSelectedModel(target.model);
      }
      if (alreadyOpen) {
        void refresh();
      }
    },
    [conversations, conversationId, refresh],
  );

  const createConversation = useCallback(async () => {
    const created = await api.createConversation(null, selectedModel);
    await loadConversations();
    selectConversation(created.conversationId);
  }, [loadConversations, selectConversation, selectedModel]);

  const handleModelChange = useCallback(
    async (model) => {
      setSelectedModel(model);
      if (conversationId) {
        await api.setConversationModel(conversationId, model);
        await loadConversations();
      }
    },
    [conversationId, loadConversations],
  );

  const deleteConversation = useCallback(
    async (id) => {
      await api.deleteConversation(id);
      const remaining = await loadConversations();
      if (id === conversationId) {
        selectConversation(remaining[0]?.id ?? null);
      }
    },
    [conversationId, loadConversations, selectConversation],
  );

  const handleSend = useCallback(
    async (text) => {
      await sendTurn(text, selectedModel, selectedModelApiKey);
      await loadConversations();
    },
    [sendTurn, loadConversations, selectedModel, selectedModelApiKey],
  );

  const activeConversation = conversations.find((conversation) => conversation.id === conversationId);
  const composerDisabled = !conversationId || status === "sending";
  const selectedConceptIds =
    turns.find((turn) => turn.id === selectedTurnId)?.conceptIds ?? [];

  return (
    <div className="appRoot">
      <ConversationSidebar
        conversations={conversations}
        conversationId={conversationId}
        onSelect={selectConversation}
        onCreate={createConversation}
        onDelete={deleteConversation}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className={`chatColumn${drawerOpen ? " chatColumnWithDrawer" : ""}`}>
        <header className="chatHeader">
          <button
            type="button"
            className="iconButton menuButton"
            aria-label={sidebarOpen ? "Hide conversation list" : "Show conversation list"}
            onClick={() => setSidebarOpen((open) => !open)}
          >
            ☰
          </button>
          <span className="chatTitle">
            {activeConversation?.title || (conversationId ? `Conversation ${conversationId}` : "AI Conversation Tree")}
          </span>
          {status === "error" ? <span className="chatError">Something went wrong</span> : null}
          <button
            type="button"
            className="ghostButton chatHeaderMap"
            onClick={() => setMapOpen(true)}
          >
            Map
          </button>
          <button
            type="button"
            className="iconButton"
            aria-label="Open settings"
            onClick={() => setSettingsOpen(true)}
          >
            ⚙
          </button>
        </header>

        <ChatTranscript
          turns={turns}
          pendingUserText={pendingUserText}
          streamingText={streamingText}
          isSending={status === "sending"}
          selectedTurnId={selectedTurnId}
          onSelectTurn={setSelectedTurnId}
          hasConversation={Boolean(conversationId)}
        />

        <Composer
          key={conversationId ?? "none"}
          onSend={handleSend}
          disabled={composerDisabled}
          placeholder={conversationId ? "Send a message" : "Start a new chat first"}
          models={availableModels}
          model={selectedModel}
          onModelChange={handleModelChange}
        />
      </main>

      {drawerOpen ? (
        <GraphDrawer
          isOpen={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          graph={graph}
          conceptLinks={conceptLinks}
          selectedConceptIds={selectedConceptIds}
          onOpenConversation={selectConversation}
          conversationId={conversationId}
          turnCount={turnCount}
          threadCount={threadCount}
          selectedTurnId={selectedTurnId}
          onSelectTurn={setSelectedTurnId}
          onAnalyze={analyze}
          onRefresh={refresh}
          isBusy={isBusy}
        />
      ) : (
        <GraphRail turnCount={turnCount} threadCount={threadCount} onOpen={() => setDrawerOpen(true)} />
      )}

      {mapOpen ? (
        <WorkspaceMap onClose={() => setMapOpen(false)} onOpenConversation={selectConversation} />
      ) : null}

      {settingsOpen ? (
        <SettingsModal
          apiKeys={apiKeys}
          onSave={saveApiKeys}
          onClose={() => setSettingsOpen(false)}
          onModelsChanged={refreshModels}
        />
      ) : null}
    </div>
  );
}
