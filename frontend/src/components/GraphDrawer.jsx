import { TurnGraph } from "./TurnGraph";

const legend = [
  { key: "continuation", label: "Continuation" },
  { key: "branch", label: "Branch" },
  { key: "related", label: "Related" },
];

export function GraphDrawer({
  isOpen,
  onClose,
  graph,
  turnCount,
  threadCount,
  selectedTurnId,
  onSelectTurn,
  onAnalyze,
  onRefresh,
  isBusy,
}) {
  return (
    <section className={`graphDrawer${isOpen ? " graphDrawerOpen" : ""}`} aria-hidden={!isOpen}>
      <header className="graphDrawerHeader">
        <div className="graphDrawerTitle">
          <span>Conversation graph</span>
          <span className="graphDrawerMeta">
            {turnCount} {turnCount === 1 ? "turn" : "turns"}
            {threadCount > 1 ? ` · ${threadCount} threads` : ""}
          </span>
        </div>
        <div className="graphDrawerActions">
          <button type="button" className="ghostButton" onClick={onRefresh} disabled={isBusy}>
            Refresh
          </button>
          <button type="button" className="ghostButton" onClick={onAnalyze} disabled={isBusy}>
            Reanalyze
          </button>
          <button type="button" className="iconButton" aria-label="Close graph" onClick={onClose}>
            ×
          </button>
        </div>
      </header>

      <div className="graphDrawerBody">
        <TurnGraph
          nodes={graph.nodes}
          edges={graph.edges}
          selectedTurnId={selectedTurnId}
          onSelectTurn={onSelectTurn}
        />
      </div>

      <footer className="graphDrawerLegend">
        {legend.map((item) => (
          <span key={item.key} className="legendItem">
            <span className={`legendSwatch legendSwatch--${item.key}`} />
            {item.label}
          </span>
        ))}
        <span className="legendItem">
          <span className="legendSwatch legendSwatch--root" />
          Root turn
        </span>
      </footer>
    </section>
  );
}
