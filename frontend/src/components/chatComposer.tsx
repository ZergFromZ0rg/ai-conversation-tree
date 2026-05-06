import { useLayoutEffect, useRef } from "react";

type ChatComposerProps = {
  value: string;
  disabled: boolean;
  onChange: (nextValue: string) => void;
  onSubmit: () => void;
};

export function ChatComposer({ value, disabled, onChange, onSubmit }: ChatComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 220)}px`;
  }, [value]);

  return (
    <div className="composerCard">
      <textarea
        ref={textareaRef}
        className="composerInput"
        value={value}
        disabled={disabled}
        rows={1}
        placeholder="Message AI Conversation Tree..."
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            onSubmit();
          }
        }}
      />

      <div className="composerFooter">
        <div className="composerHint">Enter to send · Shift+Enter for a new line</div>
        <button className="primaryButton" disabled={disabled} onClick={onSubmit}>
          {disabled ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  );
}
