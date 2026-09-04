import { useEffect, useState } from "react";
import { api } from "../api";

const PROVIDER_FIELDS = [
  { key: "openai", label: "OpenAI API key", placeholder: "sk-..." },
  { key: "anthropic", label: "Anthropic API key", placeholder: "sk-ant-..." },
  { key: "gemini", label: "Gemini API key", placeholder: "AIza..." },
];

function pullProgressLabel(progress) {
  if (!progress) {
    return null;
  }
  if (progress.status === "error") {
    return progress.message || "Download failed.";
  }
  if (progress.completed && progress.total) {
    const percent = Math.round((progress.completed / progress.total) * 100);
    return `${progress.status || "downloading"} — ${percent}%`;
  }
  return progress.status || "working…";
}

export function SettingsModal({ apiKeys, onSave, onClose, onModelsChanged }) {
  const [draft, setDraft] = useState(apiKeys);
  const [library, setLibrary] = useState(null);
  const [libraryError, setLibraryError] = useState(false);
  // Only one pull runs at a time — activePull names it (disabling other rows'
  // buttons); pullState keeps the last progress/error per model name so a
  // failed row can still show what went wrong after activePull moves on.
  const [activePull, setActivePull] = useState(null);
  const [pullState, setPullState] = useState({});

  useEffect(() => {
    setDraft(apiKeys);
  }, [apiKeys]);

  useEffect(() => {
    let cancelled = false;
    api
      .listOllamaLibrary()
      .then((payload) => {
        if (!cancelled) {
          setLibrary(payload);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLibraryError(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function handleSubmit(event) {
    event.preventDefault();
    onSave(draft);
    onClose();
  }

  async function handlePull(modelName) {
    setActivePull(modelName);
    setPullState((current) => ({ ...current, [modelName]: null }));
    try {
      for await (const progress of api.pullOllamaModel(modelName)) {
        setPullState((current) => ({ ...current, [modelName]: progress }));
        if (progress.status === "error") {
          setActivePull(null);
          return;
        }
      }
      setLibrary((current) =>
        current ? { ...current, models: current.models.filter((entry) => entry.name !== modelName) } : current,
      );
      setPullState((current) => ({ ...current, [modelName]: undefined }));
      await onModelsChanged?.();
    } catch {
      setPullState((current) => ({ ...current, [modelName]: { status: "error", message: "Download failed." } }));
    } finally {
      setActivePull((current) => (current === modelName ? null : current));
    }
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

          <div className="settingsDivider" />

          <div className="settingsSection">
            <span className="settingsSectionTitle">Download a local model (Ollama)</span>
            {libraryError ? (
              <p className="settingsHint">Couldn&rsquo;t reach this app&rsquo;s server to check Ollama.</p>
            ) : !library ? (
              <p className="settingsHint">Checking Ollama…</p>
            ) : !library.ollamaReachable ? (
              <p className="settingsHint">
                Ollama isn&rsquo;t running. Install it from{" "}
                <a href="https://ollama.com" target="_blank" rel="noreferrer">
                  ollama.com
                </a>{" "}
                and start it, then reopen Settings to download a model here.
              </p>
            ) : library.models.length === 0 ? (
              <p className="settingsHint">Every model in the curated list below is already installed.</p>
            ) : (
              <ul className="ollamaLibraryList">
                {library.models.map((entry) => {
                  const isPulling = activePull === entry.name;
                  const progress = pullState[entry.name];
                  const hasFailed = progress?.status === "error";
                  return (
                    <li key={entry.name} className="ollamaLibraryRow">
                      <span className="ollamaLibraryName">{entry.name}</span>
                      {isPulling || hasFailed ? (
                        <span className={`ollamaLibraryProgress${hasFailed ? " ollamaLibraryProgressError" : ""}`}>
                          {pullProgressLabel(progress)}
                        </span>
                      ) : (
                        <span className="ollamaLibrarySize">{entry.sizeLabel}</span>
                      )}
                      <button
                        type="button"
                        className="ghostButton ollamaLibraryButton"
                        disabled={isPulling || (activePull !== null && !hasFailed)}
                        onClick={() => handlePull(entry.name)}
                      >
                        {isPulling ? "Downloading…" : hasFailed ? "Retry" : "Download"}
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

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
