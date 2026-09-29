import { useEffect, useState } from "react";
import { api } from "../api";

export function AnalysisPanel({ mode, conversationId, onClose, onOpenConversation }) {
  const [query, setQuery] = useState("");
  const [summary, setSummary] = useState(null);
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    if (mode !== "summary" || !conversationId) return;
    setStatus("loading");
    setError("");
    api.getSummary(conversationId).then(setSummary).catch((err) => setError(err.message)).finally(() => setStatus("idle"));
  }, [mode, conversationId]);

  async function runSearch(event) {
    event.preventDefault();
    if (!query.trim()) return;
    setStatus("loading");
    setError("");
    try {
      const payload = await api.searchConversations(query);
      setResults(payload.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setStatus("idle");
    }
  }

  return (
    <div className="analysisBackdrop" onClick={onClose}>
      <section className="analysisPanel" onClick={(event) => event.stopPropagation()}>
        <div className="analysisHeader">
          <h2>{mode === "search" ? "Search conversations" : "Conversation summary"}</h2>
          <button type="button" className="iconButton" onClick={onClose} aria-label="Close analysis">×</button>
        </div>
        {mode === "search" ? (
          <>
            <form className="analysisSearchForm" onSubmit={runSearch}>
              <input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search across all chats" />
              <button type="submit" className="primaryButton" disabled={status === "loading"}>Search</button>
            </form>
            {results.length === 0 && status !== "loading" ? <p className="analysisMuted">Search for a topic, question, or idea.</p> : null}
            <div className="analysisResults">
              {results.map((result) => (
                <button type="button" className="analysisResult" key={`${result.conversationId}-${result.turnId}`} onClick={() => { onOpenConversation(result.conversationId); onClose(); }}>
                  <span className="analysisResultTitle">{result.conversationTitle || `Conversation ${result.conversationId}`}</span>
                  <span className="analysisResultText">{result.userText}</span>
                  <span className="analysisResultScore">match {Math.round(result.score * 100)}%</span>
                </button>
              ))}
            </div>
          </>
        ) : (
          <div className="analysisSummary">
            {status === "loading" ? <p className="analysisMuted">Building summary…</p> : null}
            {error ? <p className="chatError">{error}</p> : null}
            {summary ? (
              <>
                <p>{summary.summary}</p>
                <div className="analysisStats">
                  <span>{summary.turnCount} turns</span>
                  <span>{summary.threadCount} threads</span>
                  <span>{summary.edgeCount} links</span>
                </div>
                <p className="analysisTopics"><strong>Topics:</strong> {summary.topics.join(", ") || "None detected"}</p>
              </>
            ) : null}
          </div>
        )}
      </section>
    </div>
  );
}
