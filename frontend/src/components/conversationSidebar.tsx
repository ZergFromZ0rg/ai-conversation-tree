import type { ConversationSummary } from "../types";

type ConversationSidebarProps = {
  conversations: ConversationSummary[];
  activeConversationId: number | null;
  isOpen: boolean;
  onClose: () => void;
  onCreateConversation: () => void;
  onSelectConversation: (conversationId: number) => void;
};

export function ConversationSidebar({
  conversations,
  activeConversationId,
  isOpen,
  onClose,
  onCreateConversation,
  onSelectConversation
}: ConversationSidebarProps) {
  return (
    <aside className={`drawer leftDrawer ${isOpen ? "open" : ""}`}>
      <div className="drawerHeader">
        <div className="drawerTitle">Chats</div>
        <button className="iconButton" onClick={onClose}>✕</button>
      </div>
      <div className="drawerBody">
        <button className="primaryButton fullWidthButton" onClick={onCreateConversation}>
          + New chat
        </button>

        <div className="conversationList">
          {conversations.map((conversation) => (
            <button
              key={conversation.id}
              className={`conversationItem ${activeConversationId === conversation.id ? "active" : ""}`}
              onClick={() => onSelectConversation(conversation.id)}
            >
              <div className="conversationName">{conversation.title || `Conversation ${conversation.id}`}</div>
              <div className="conversationMeta">Updated {conversation.updatedAt}</div>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}
