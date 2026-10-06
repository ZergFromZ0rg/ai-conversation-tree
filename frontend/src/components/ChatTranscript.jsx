import { useEffect, useRef } from "react";
import Markdown from "react-markdown";

function ChatMessage({ role, text, onSelect, isSelected }) {
  return (
    <div
      className={`message message--${role}${isSelected ? " message--selected" : ""}`}
      onClick={onSelect}
    >
      <div className="messageRole">{role === "user" ? "You" : "Assistant"}</div>
      {text ? (
        role === "assistant" ? (
          <div className="messageText messageMarkdown">
            <Markdown>{text}</Markdown>
          </div>
        ) : (
          <div className="messageText">{text}</div>
        )
      ) : (
        <div className="messageText">
          <span className="messageMuted">…</span>
        </div>
      )}
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

function ContextFlag({ flag }) {
  const detail = flag.signals.map((signal) => signal.detail).join(" · ");
  return (
    <div className={`contextFlag contextFlag--${flag.level}`} title={detail}>
      <span aria-hidden="true">⚠</span> Possible context loss — {flag.signals[0].detail}
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
  contextHealth,
  hasConversation,
}) {
  const turnRefs = useRef(new Map());
  const bottomRef = useRef(null);
  // A turn selected before its conversation finished loading (a search hit in
  // another chat) can't be scrolled to yet; remember it and finish the jump
  // when the turns arrive.
  const pendingJumpRef = useRef(null);

  useEffect(() => {
    // Instant while text is actively growing (a "smooth" scroll per token
    // would fight itself); smooth for turn/state changes otherwise. Skipped
    // while a jump to a specific turn is outstanding. (Declared before the
    // jump effect so it still sees the pending jump on the render that
    // delivers the turns.)
    if (pendingJumpRef.current != null) {
      return;
    }
    bottomRef.current?.scrollIntoView({ behavior: streamingText ? "auto" : "smooth", block: "end" });
  }, [turns.length, pendingUserText, isSending, streamingText]);

  useEffect(() => {
    pendingJumpRef.current = selectedTurnId;
  }, [selectedTurnId]);

  useEffect(() => {
    const target = pendingJumpRef.current;
    if (target == null) {
      return;
    }
    const node = turnRefs.current.get(target);
    if (node) {
      node.scrollIntoView({ behavior: "smooth", block: "start" });
      pendingJumpRef.current = null;
    }
  }, [selectedTurnId, turns]);

  const flagsByTurn = new Map(
    (contextHealth?.flaggedTurns ?? [])
      .filter((flag) => flag.level !== "low")
      .map((flag) => [flag.turnId, flag]),
  );

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
            {turn.id === contextHealth?.horizonTurnId ? (
              <div className="contextHorizon">
                <span>
                  Earlier messages are probably outside the model&rsquo;s context window
                  (~{contextHealth.windowTokens.toLocaleString()} tokens)
                </span>
              </div>
            ) : null}
            <ChatMessage
              role="user"
              text={turn.userText}
              isSelected={turn.id === selectedTurnId}
              onSelect={() => onSelectTurn(turn.id)}
            />
            {flagsByTurn.has(turn.id) ? <ContextFlag flag={flagsByTurn.get(turn.id)} /> : null}
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
