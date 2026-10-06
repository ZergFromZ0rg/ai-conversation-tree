import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";

const SOURCE_LABELS = { chatgpt: "ChatGPT", claude: "Claude", gemini: "Gemini" };

function Highlight({ text, terms }) {
  if (!terms?.length) return text;
  // Whole words around each term, so "index" lights up all of "indexes".
  const escaped = terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`(\\w*(?:${escaped.join("|")})\\w*)`, "gi");
  return text.split(pattern).map((part, index) =>
    index % 2 === 1 ? <mark key={index}>{part}</mark> : <Fragment key={index}>{part}</Fragment>,
  );
}

function groupByConversation(results) {
  const groups = new Map();
  for (const result of results) {
    if (!groups.has(result.conversationId)) {
      groups.set(result.conversationId, {
        conversationId: result.conversationId,
        title: result.conversationTitle || `Conversation ${result.conversationId}`,
        source: result.source,
        hits: [],
      });
    }
    groups.get(result.conversationId).hits.push(result);
  }
  return Array.from(groups.values());
}

function SearchView({ conversationId, onOpenTurn, onClose }) {
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState("all");
  const [payload, setPayload] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [active, setActive] = useState(0);
  const activeRef = useRef(null);

  // Search as you type (debounced) — there is no separate submit step.
  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setPayload(null);
      setStatus("idle");
      return undefined;
    }
    let cancelled = false;
    setStatus("loading");
    const timer = setTimeout(() => {
      api
        .searchConversations(trimmed, 30, scope === "this" ? conversationId : null)
        .then((next) => {
          if (cancelled) return;
          setPayload(next);
          setActive(0);
          setError("");
        })
        .catch((err) => !cancelled && setError(err.message))
        .finally(() => !cancelled && setStatus("idle"));
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [query, scope, conversationId]);

  const groups = useMemo(() => groupByConversation(payload?.results ?? []), [payload]);
  const flat = useMemo(() => groups.flatMap((group) => group.hits), [groups]);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "nearest" });
  }, [active, payload]);

  function open(hit) {
    onOpenTurn(hit.conversationId, hit.turnId);
    onClose();
  }

  function onKeyDown(event) {
    if (event.key === "ArrowDown" && flat.length) {
      event.preventDefault();
      setActive((index) => Math.min(index + 1, flat.length - 1));
    } else if (event.key === "ArrowUp" && flat.length) {
      event.preventDefault();
      setActive((index) => Math.max(index - 1, 0));
    } else if (event.key === "Enter" && flat[active]) {
      event.preventDefault();
      open(flat[active]);
    }
  }

  let position = -1;
  return (
    <>
      <div className="analysisSearchForm">
        <input
          autoFocus
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Search every chat — a topic, a question, a phrase"
          aria-label="Search conversations"
        />
      </div>
      <div className="searchScope" role="group" aria-label="Search scope">
        <button type="button" className={scope === "all" ? "active" : ""} onClick={() => setScope("all")}>
          All chats
        </button>
        <button
          type="button"
          className={scope === "this" ? "active" : ""}
          onClick={() => setScope("this")}
          disabled={!conversationId}
        >
          This chat
        </button>
        <span className="searchHint">↑↓ to move · Enter to jump to the turn</span>
      </div>

      {error ? <p className="chatError">{error}</p> : null}
      {!payload && status !== "loading" ? (
        <p className="analysisMuted">Finds turns by meaning and by exact words, in questions and answers.</p>
      ) : null}
      {payload && flat.length === 0 && status !== "loading" ? (
        <p className="analysisMuted">No turns match “{payload.query}”.</p>
      ) : null}

      <div className="searchGroups">
        {groups.map((group) => (
          <section className="searchGroup" key={group.conversationId}>
            <h3>
              <span>{group.title}</span>
              {group.source ? <span className="sourceBadge">{SOURCE_LABELS[group.source] ?? group.source}</span> : null}
              <span className="searchGroupCount">
                {group.hits.length} match{group.hits.length === 1 ? "" : "es"}
              </span>
            </h3>
            {group.hits.map((hit) => {
              position += 1;
              const isActive = position === active;
              return (
                <button
                  type="button"
                  key={`${hit.conversationId}-${hit.turnId}`}
                  ref={isActive ? activeRef : null}
                  className={`searchHit${isActive ? " searchHitActive" : ""}`}
                  onClick={() => open(hit)}
                  onMouseMove={() => setActive(flat.indexOf(hit))}
                >
                  <span className="searchHitMeta">
                    <span className={`matchTag matchTag--${hit.matchIn}`}>{hit.matchIn}</span>
                    <span>
                      turn {hit.turnId + 1} of {hit.turnCount}
                    </span>
                    <span className="searchHitScore">{Math.round(hit.score * 100)}%</span>
                  </span>
                  <span className="searchHitSnippet">
                    <Highlight text={hit.snippet} terms={hit.matchedTerms} />
                  </span>
                </button>
              );
            })}
          </section>
        ))}
      </div>
    </>
  );
}

const LEVEL_LABELS = { low: "Healthy", medium: "Watch", high: "At risk" };

function ContextHealth({ conversationId, onOpenTurn, onClose }) {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    setHealth(null);
    api.getContextHealth(conversationId).then(setHealth).catch(() => setHealth(null));
  }, [conversationId]);

  if (!health || health.turnCount === 0) return null;
  const flagged = health.flaggedTurns.filter((item) => item.level !== "low");
  return (
    <div className="contextHealth">
      <h3>
        Context health
        <span className={`riskPill riskPill--${health.overall}`}>{LEVEL_LABELS[health.overall]}</span>
      </h3>
      <p className="analysisMuted">
        ~{health.estimatedTokens.toLocaleString()} tokens so far · {Math.round(health.windowPressure * 100)}% of an
        assumed {health.windowTokens.toLocaleString()}-token window
        {health.horizonTurnId != null ? ` · turns before #${health.horizonTurnId + 1} are likely out of view` : ""}
      </p>
      {flagged.length === 0 ? (
        <p className="analysisMuted">No signs of lost context: no repeated questions, “as I said” callbacks, or long-range dependencies.</p>
      ) : (
        <ul>
          {flagged.map((item) => (
            <li key={item.turnId}>
              <button
                type="button"
                onClick={() => {
                  onOpenTurn(health.conversationId, item.turnId);
                  onClose();
                }}
              >
                <span className={`riskPill riskPill--${item.level}`}>turn {item.turnId + 1}</span>
                <span>{item.signals.map((signal) => signal.detail).join(" · ")}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SummaryView({ conversationId, model, apiKey, onOpenTurn, onClose }) {
  const [summary, setSummary] = useState(null);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!conversationId) return;
    setStatus("loading");
    setError("");
    api
      .getSummary(conversationId)
      .then(setSummary)
      .catch((err) => setError(err.message))
      .finally(() => setStatus("idle"));
  }, [conversationId]);

  async function writeWithModel() {
    setStatus("writing");
    setError("");
    try {
      setSummary(await api.getLlmSummary(conversationId, model, apiKey));
    } catch (err) {
      setError(err.message);
    } finally {
      setStatus("idle");
    }
  }

  return (
    <div className="analysisSummary">
      {status === "loading" ? <p className="analysisMuted">Building summary…</p> : null}
      {error ? <p className="chatError">{error}</p> : null}
      {summary ? (
        <>
          <p>{summary.summary}</p>
          <p className="analysisMuted">
            {summary.mode === "llm" ? `Written by ${summary.model}` : "Structured local summary"}
            {summary.fallbackReason ? ` — ${summary.fallbackReason}` : ""}
          </p>
          <div className="analysisStats">
            <span>{summary.turnCount} turns</span>
            <span>{summary.threadCount} threads</span>
            <span>{summary.edgeCount} links</span>
          </div>
          <p className="analysisTopics">
            <strong>Topics:</strong> {summary.topics.join(", ") || "None detected"}
          </p>
          <button type="button" className="ghostButton" onClick={writeWithModel} disabled={status === "writing" || summary.turnCount === 0}>
            {status === "writing" ? "Writing…" : summary.mode === "llm" ? "Rewrite with model" : "Write with the selected model"}
          </button>
          <ContextHealth conversationId={conversationId} onOpenTurn={onOpenTurn} onClose={onClose} />
        </>
      ) : null}
    </div>
  );
}

export function AnalysisPanel({ mode, conversationId, model, apiKey, onClose, onOpenTurn }) {
  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div className="analysisBackdrop" onClick={onClose}>
      <section className="analysisPanel" onClick={(event) => event.stopPropagation()}>
        <div className="analysisHeader">
          <h2>{mode === "search" ? "Search conversations" : "Conversation summary"}</h2>
          <button type="button" className="iconButton" onClick={onClose} aria-label="Close analysis">
            ×
          </button>
        </div>
        {mode === "search" ? (
          <SearchView conversationId={conversationId} onOpenTurn={onOpenTurn} onClose={onClose} />
        ) : (
          <SummaryView
            conversationId={conversationId}
            model={model}
            apiKey={apiKey}
            onOpenTurn={onOpenTurn}
            onClose={onClose}
          />
        )}
      </section>
    </div>
  );
}
