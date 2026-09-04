import { useEffect, useRef } from "react";

function ChatMessage({ role, text, onSelect, isSelected }) {
  return (
    <div
      className={`message message--${role}${isSelected ? " message--selected" : ""}`}
      onClick={onSelect}
    >
      <div className="messageRole">{role === "user" ? "You" : "Assistant"}</div>
      <div className="messageText">{text || <span className="messageMuted">…</span>}</div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="message message--assistant">
      <div className="messageRole">Assistant</div>
      <div className="typingDots" aria-label="Assistant is replying">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}

export function ChatTranscript({
  turns,
  pendingUserText,
  isSending,
  streamingText,
  selectedTurnId,
  onSelectTurn,
  hasConversation,
}) {
  const turnRefs = useRef(new Map());
  const bottomRef = useRef(null);

  useEffect(() => {
    if (selectedTurnId == null) {
      return;
    }
    const node = turnRefs.current.get(selectedTurnId);
    if (node) {
      node.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [selectedTurnId]);

  useEffect(() => {
    // Instant while text is actively growing (a "smooth" scroll per token
    // would fight itself); smooth for turn/state changes otherwise.
    bottomRef.current?.scrollIntoView({ behavior: streamingText ? "auto" : "smooth", block: "end" });
  }, [turns.length, pendingUserText, isSending, streamingText]);

  if (!hasConversation) {
    return (
      <div className="transcript transcriptEmpty">
        <p>Pick a conversation or start a new chat.</p>
      </div>
    );
  }

  if (turns.length === 0 && !pendingUserText) {
    return (
      <div className="transcript transcriptEmpty">
        <p>Send a message to start the conversation.</p>
      </div>
    );
  }

  return (
    <div className="transcript">
      <div className="transcriptInner">
        {turns.map((turn) => (
          <div
            key={turn.id}
            className="turnBlock"
            ref={(node) => {
              if (node) {
                turnRefs.current.set(turn.id, node);
              } else {
                turnRefs.current.delete(turn.id);
              }
            }}
          >
            <ChatMessage
              role="user"
              text={turn.userText}
              isSelected={turn.id === selectedTurnId}
              onSelect={() => onSelectTurn(turn.id)}
            />
            <ChatMessage
              role="assistant"
              text={turn.aiText}
              isSelected={turn.id === selectedTurnId}
              onSelect={() => onSelectTurn(turn.id)}
            />
          </div>
        ))}

        {pendingUserText ? (
          <div className="turnBlock">
            <ChatMessage role="user" text={pendingUserText} />
            {streamingText ? (
              <ChatMessage role="assistant" text={streamingText} />
            ) : isSending ? (
              <TypingIndicator />
            ) : null}
          </div>
        ) : null}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
