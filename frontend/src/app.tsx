import { useEffect, useMemo, useState } from "react";
import { ChatComposer } from "./components/chatComposer";
import { ChatThread } from "./components/chatThread";
import { ConversationSidebar } from "./components/conversationSidebar";
import { GraphDrawer } from "./components/graphDrawer";
import { createConversation, getConversation, listConversations, reclassifyGraph, sendChatMessage } from "./lib/api";
import type { ConversationPayload, ConversationSummary } from "./types";

const emptyConversation: ConversationPayload = {
  conversationId: 0,
  turns: [],
  nodes: [],
  edges: []
};

export function App() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [conversation, setConversation] = useState<ConversationPayload>(emptyConversation);
  const [message, setMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(typeof window !== "undefined" ? window.innerWidth >= 900 : true);
  const [isGraphOpen, setIsGraphOpen] = useState(false);
  const [toast, setToast] = useState<{ type: "success" | "error"; message: string } | null>(null);

  function showToast(type: "success" | "error", nextMessage: string) {
    setToast({ type, message: nextMessage });
    window.clearTimeout((showToast as typeof showToast & { timeoutId?: number }).timeoutId);
    (showToast as typeof showToast & { timeoutId?: number }).timeoutId = window.setTimeout(() => {
      setToast(null);
    }, 3200);
  }

  async function refreshConversationList() {
    const nextConversations = await listConversations();
    setConversations(nextConversations);
  }

  async function ensureConversation(): Promise<number> {
    if (activeConversationId !== null) {
      return activeConversationId;
    }

    const conversationId = await createConversation("New Conversation");
    setActiveConversationId(conversationId);
    return conversationId;
  }

  async function loadConversation(conversationId: number) {
    const payload = await getConversation(conversationId);
    setActiveConversationId(conversationId);
    setConversation(payload);
  }

  async function handleCreateConversation() {
    const conversationId = await createConversation("New Conversation");
    setActiveConversationId(conversationId);
    setConversation({ ...emptyConversation, conversationId });
    setMessage("");
    await refreshConversationList();
    showToast("success", `Created conversation ${conversationId}.`);
  }

  async function handleSelectConversation(conversationId: number) {
    await loadConversation(conversationId);
    if (window.innerWidth < 900) {
      setIsHistoryOpen(false);
    }
  }

  async function handleSendMessage() {
    const trimmedMessage = message.trim();
    if (!trimmedMessage || isSending) {
      return;
    }

    try {
      setIsSending(true);
      const conversationId = await ensureConversation();
      const payload = await sendChatMessage(conversationId, trimmedMessage);
      setActiveConversationId(payload.conversationId);
      setConversation(payload);
      setMessage("");
      await refreshConversationList();
      showToast("success", `Turn ${payload.turnId} added.`);
    } catch (error) {
      console.error(error);
      showToast("error", "Error sending message.");
    } finally {
      setIsSending(false);
    }
  }

  async function handleRefreshGraph() {
    if (activeConversationId === null) {
      return;
    }
    try {
      await loadConversation(activeConversationId);
    } catch (error) {
      console.error(error);
      showToast("error", "Error loading conversation.");
    }
  }

  async function handleReclassify() {
    try {
      const payload = await reclassifyGraph();
      if (activeConversationId !== null) {
        await loadConversation(activeConversationId);
      }
      showToast("success", `Reclassified ${payload.nodes} turns and ${payload.edges} edges.`);
    } catch (error) {
      console.error(error);
      showToast("error", "Error reclassifying graph.");
    }
  }

  useEffect(() => {
    async function initialize() {
      const existingConversations = await listConversations();
      setConversations(existingConversations);

      if (existingConversations.length > 0) {
        const conversationId = existingConversations[0].id;
        setActiveConversationId(conversationId);
        const payload = await getConversation(conversationId);
        setConversation(payload);
        return;
      }

      const conversationId = await createConversation("UI Conversation");
      setActiveConversationId(conversationId);
      const payload = await getConversation(conversationId);
      setConversation(payload);
      await refreshConversationList();
    }

    void initialize();
  }, []);

  const activeConversationLabel = useMemo(() => {
    if (activeConversationId === null) {
      return "No active conversation";
    }
    const count = conversation.turns.length;
    return `Conversation ${activeConversationId} · ${count} turn${count === 1 ? "" : "s"}`;
  }, [activeConversationId, conversation.turns.length]);

  return (
    <>
      <div
        className={`backdrop ${(isHistoryOpen || isGraphOpen) ? "visible" : ""}`}
        onClick={() => {
          setIsHistoryOpen(false);
          setIsGraphOpen(false);
        }}
      />

      <div className="appShell">
        <ConversationSidebar
          conversations={conversations}
          activeConversationId={activeConversationId}
          isOpen={isHistoryOpen}
          onClose={() => setIsHistoryOpen(false)}
          onCreateConversation={handleCreateConversation}
          onSelectConversation={handleSelectConversation}
        />

        <main className="mainPanel">
          <div className="topBar">
            <div className="topBarGroup">
              <button className="iconButton" onClick={() => setIsHistoryOpen((value) => !value)}>☰</button>
              <div className="titleBlock">
                <h1>AI Conversation Tree</h1>
                <div className="titleMeta">{activeConversationLabel}</div>
              </div>
            </div>

            <div className="topBarGroup">
              <button onClick={handleCreateConversation}>New Chat</button>
              <button onClick={() => setIsGraphOpen((value) => !value)}>Graph</button>
            </div>
          </div>

          <section className="chatViewport">
            <div className="chatColumn">
              <ChatThread turns={conversation.turns} />
            </div>
          </section>

          <div className="composerWrap">
            <ChatComposer
              value={message}
              disabled={isSending}
              onChange={setMessage}
              onSubmit={() => void handleSendMessage()}
            />
          </div>
        </main>

        <GraphDrawer
          isOpen={isGraphOpen}
          nodes={conversation.nodes}
          edges={conversation.edges}
          onClose={() => setIsGraphOpen(false)}
          onRefresh={() => void handleRefreshGraph()}
          onReclassify={() => void handleReclassify()}
        />
      </div>

      {toast ? <div className={`statusToast ${toast.type}`}>{toast.message}</div> : null}
    </>
  );
}
