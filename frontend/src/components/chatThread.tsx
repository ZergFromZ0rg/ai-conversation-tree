import type { Turn } from "../types";

type ChatThreadProps = {
  turns: Turn[];
};

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;")
    .replace(/\n/g, "<br>");
}

export function ChatThread({ turns }: ChatThreadProps) {
  if (!turns.length) {
    return (
      <div className="emptyState">
        <h2>Start a conversation</h2>
        <p>The center panel stays linear. Open the graph drawer on the right when you want structure.</p>
      </div>
    );
  }

  return (
    <div className="messages">
      {turns.map((turn) => (
        <div key={turn.id} className="turnGroup">
          <div className="messageRow user">
            <div className="messageBubble userBubble">
              <div dangerouslySetInnerHTML={{ __html: escapeHtml(turn.userText) }} />
            </div>
          </div>

          <div className="messageRow assistant">
            <div className="assistantBlock">
              <div className="messageLabel">Assistant</div>
              <div className="messageBubble assistantBubble">
                <div dangerouslySetInnerHTML={{ __html: escapeHtml(turn.aiText) }} />
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
