import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { useConversation } from "./useConversation";
import { ConversationSidebar } from "./components/ConversationSidebar";
import { ChatTranscript } from "./components/ChatTranscript";
import { Composer } from "./components/Composer";
import { GraphRail } from "./components/GraphRail";
import { GraphDrawer } from "./components/GraphDrawer";
import { WorkspaceMap } from "./components/WorkspaceMap";

const DRAWER_STORAGE_KEY = "act.drawerOpen";
const MODEL_STORAGE_KEY = "act.model";

function readDrawerPreference() {
  try {
    return window.localStorage.getItem(DRAWER_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function readModelPreference() {
  try {
    return window.localStorage.getItem(MODEL_STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

export function App() {
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [selectedTurnId, setSelectedTurnId] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(readDrawerPreference);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mapOpen, setMapOpen] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(readModelPreference);

  const { turns, graph, conceptLinks, status, pendingUserText, sendTurn, analyze, refresh } =
    useConversation(conversationId);

  const isBusy = status === "loading" || status === "sending" || status === "analyzing";
  const turnCount = turns.length;
  const threadCount = turns.filter((turn) => turn.root).length;

  const loadConversations = useCallback(async () => {
    const payload = await api.listConversations();
    setConversations(payload.conversations);
    return payload.conversations;
  }, []);

  useEffect(() => {
    void loadConversations().then((list) => {
      setConversationId((current) => current ?? list[0]?.id ?? null);
    });
  }, [loadConversations]);

  useEffect(() => {
    void api
      .listModels()
      .then((payload) => {
        setModels(payload.models);
        setSelectedModel((current) => current ?? payload.default ?? null);
      })
      .catch(() => setModels([]));
  }, []);

  useEffect(() => {
    try {
      if (selectedModel) {
        window.localStorage.setItem(MODEL_STORAGE_KEY, selectedModel);
      }
    } catch {
      /* storage unavailable — ignore */
    }
  }, [selectedModel]);

  useEffect(() => {
    try {
      window.localStorage.setItem(DRAWER_STORAGE_KEY, drawerOpen ? "1" : "0");
    } catch {
      /* storage unavailable — ignore */
    }
  }, [drawerOpen]);

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
      setSidebarOpen(false);
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
      await sendTurn(text, selectedModel);
      await loadConversations();
    },
    [sendTurn, loadConversations, selectedModel],
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
            aria-label="Open menu"
            onClick={() => setSidebarOpen(true)}
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
        </header>

        <ChatTranscript
          turns={turns}
          pendingUserText={pendingUserText}
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
          models={models}
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
    </div>
  );
}
