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
}) {
  return (
    <>
      <div className={`sidebarBackdrop${isOpen ? " sidebarBackdropVisible" : ""}`} onClick={onClose} />
      <aside className={`sidebar${isOpen ? " sidebarOpen" : ""}`}>
        <div className="sidebarHeader">
          <span className="sidebarBrand">
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
          <button type="button" className="iconButton sidebarCloseButton" aria-label="Close menu" onClick={onClose}>
            ×
          </button>
        </div>

        <button type="button" className="newChatButton" onClick={onCreate}>
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
      </aside>
    </>
  );
}
