import { useEffect, useState } from "react";

const PROVIDER_FIELDS = [
  { key: "openai", label: "OpenAI API key", placeholder: "sk-..." },
  { key: "anthropic", label: "Anthropic API key", placeholder: "sk-ant-..." },
  { key: "gemini", label: "Gemini API key", placeholder: "AIza..." },
];

export function SettingsModal({ apiKeys, onSave, onClose }) {
  const [draft, setDraft] = useState(apiKeys);

  useEffect(() => {
    setDraft(apiKeys);
  }, [apiKeys]);

  function handleSubmit(event) {
    event.preventDefault();
    onSave(draft);
    onClose();
  }

  return (
    <div className="workspaceMapBackdrop" onClick={onClose}>
      <div className="settingsModal" onClick={(event) => event.stopPropagation()}>
        <header className="workspaceMapHeader">
          <div className="graphDrawerTitle">
            <span>Settings</span>
          </div>
          <button type="button" className="iconButton" aria-label="Close settings" onClick={onClose}>
            ×
          </button>
        </header>

        <form className="settingsForm" onSubmit={handleSubmit}>
          <p className="settingsHint">
            Ollama runs free and local, no key needed — it&rsquo;s picked up automatically when
            reachable. Keys entered here are stored only in this browser (localStorage) and sent
            to this app&rsquo;s own local server only when you send a message with that model
            selected. They are never written to disk on the server or saved anywhere else.
          </p>
          {PROVIDER_FIELDS.map((field) => (
            <label key={field.key} className="settingsField">
              <span>{field.label}</span>
              <input
                type="password"
                autoComplete="off"
                spellCheck={false}
                value={draft[field.key] ?? ""}
                placeholder={field.placeholder}
                onChange={(event) =>
                  setDraft((current) => ({ ...current, [field.key]: event.target.value }))
                }
              />
            </label>
          ))}
          <div className="settingsActions">
            <button type="button" className="ghostButton" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="newChatButton settingsSave">
              Save
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
