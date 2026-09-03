import { useEffect, useRef, useState } from "react";

const MAX_HEIGHT = 200;

export function Composer({ onSend, disabled, placeholder }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    // Empty: let CSS size it to one row (measuring scrollHeight in an
    // unlaid-out / hidden context can otherwise stick it at the max height).
    if (!value) {
      textarea.style.height = "";
      return;
    }
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_HEIGHT)}px`;
  }, [value]);

  function submit() {
    const text = value.trim();
    if (!text || disabled) {
      return;
    }
    onSend(text);
    setValue("");
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="composer">
      <div className="composerBox">
        <textarea
          ref={textareaRef}
          className="composerInput"
          rows={1}
          value={value}
          disabled={disabled}
          placeholder={placeholder}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          type="button"
          className="composerSend"
          aria-label="Send message"
          disabled={disabled || !value.trim()}
          onClick={submit}
        >
          ↑
        </button>
      </div>
      <p className="composerHint">Enter to send · Shift+Enter for a new line</p>
    </div>
  );
}
