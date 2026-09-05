import { useState } from "react";

function ConversationRow({ conversation, isActive, onSelect, onDelete }) {
  const [confirming, setConfirming] = useState(false);
  const title = conversation.title || `Conversation ${conversation.id}`;

  return (
    <div className={`conversationRow${isActive ? " conversationRowActive" : ""}`}>
      <button type="button" className="conversationRowSelect" onClick={() => onSelect(conversation.id)}>
        <span className="conversationRowTitle">{title}</span>
      </button>
      {confirming ? (
        <span className="conversationRowConfirm">
          <button type="button" className="linkDanger" onClick={() => onDelete(conversation.id)}>
            delete
          </button>
          <button type="button" className="linkMuted" onClick={() => setConfirming(false)}>
            keep
          </button>
        </span>
      ) : (
        <button
          type="button"
          className="conversationRowDelete"
          aria-label={`Delete ${title}`}
          onClick={() => setConfirming(true)}
        >
          ×
        </button>
      )}
    </div>
  );
}

export function ConversationSidebar({
  conversations,
  conversationId,
  onSelect,
  onCreate,
  onDelete,
  isOpen,
  onClose,
  onToggle,
}) {
  return (
    <>
      <div className={`sidebarBackdrop${isOpen ? " sidebarBackdropVisible" : ""}`} onClick={onClose} />
      <aside className={`sidebar${isOpen ? " sidebarOpen" : " sidebarCollapsed"}`}>
        <div className="sidebarHeader">
          {/* First in the DOM so it stays visible in the collapsed strip —
              justify-content: space-between keeps it flush left, and the
              sidebar's own overflow: hidden clips everything after it once
              the sidebar narrows. */}
          <button
            type="button"
            className="iconButton sidebarToggleButton"
            aria-label={isOpen ? "Hide conversation list" : "Show conversation list"}
            onClick={onToggle}
          >
            ☰
          </button>
          <span className="sidebarBrand" aria-hidden={!isOpen}>
            <svg
              className="sidebarBrandMark"
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
              aria-hidden="true"
            >
              <circle cx="8" cy="2.5" r="1.6" fill="currentColor" />
              <circle cx="3" cy="13.5" r="1.6" fill="currentColor" />
              <circle cx="13" cy="13.5" r="1.6" fill="currentColor" />
              <path
                d="M8 4.1V8M8 8L3 11.9M8 8l5 3.9"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
              />
            </svg>
            AI Conversation Tree
          </span>
        </div>

        <div className="sidebarContent" aria-hidden={!isOpen}>
          <button type="button" className="newChatButton" onClick={onCreate} tabIndex={isOpen ? 0 : -1}>
            <span aria-hidden="true">＋</span> New chat
          </button>

          <div className="conversationList">
            {conversations.length === 0 ? (
              <p className="sidebarEmpty">No conversations yet.</p>
            ) : (
              conversations.map((conversation) => (
                <ConversationRow
                  key={conversation.id}
                  conversation={conversation}
                  isActive={conversation.id === conversationId}
                  onSelect={onSelect}
                  onDelete={onDelete}
                />
              ))
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
