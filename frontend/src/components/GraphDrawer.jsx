import { TurnGraph } from "./TurnGraph";

const legend = [
  { key: "continuation", label: "Continuation" },
  { key: "branch", label: "Branch" },
  { key: "related", label: "Related" },
];

function ConceptLinksPanel({ concepts, selectedConceptIds, onOpenConversation }) {
  const hasSelection = selectedConceptIds.length > 0;
  const groups = hasSelection
    ? concepts.filter((concept) => selectedConceptIds.includes(concept.conceptId))
    : concepts;

  if (concepts.length === 0) {
    return null;
  }

  return (
    <section className="conceptLinks">
      <div className="conceptLinksTitle">Also discussed elsewhere</div>
      {groups.length === 0 ? (
        <p className="conceptLinksHint">This turn&rsquo;s topics aren&rsquo;t linked to other chats.</p>
      ) : (
        groups.map((group) => (
          <div key={group.conceptId} className="conceptLinkGroup">
            <div className="conceptLinkConcept">{group.label || `Concept ${group.conceptId}`}</div>
            <ul className="conceptLinkList">
              {group.links.map((link) => (
                <li key={`${link.conversationId}-${link.conceptId}`}>
                  <button
                    type="button"
                    className="conceptLinkItem"
                    onClick={() => onOpenConversation(link.conversationId)}
                  >
                    <span className={`conceptLinkKind conceptLinkKind--${link.kind}`}>{link.kind}</span>
                    <span className="conceptLinkText">
                      {link.conversationTitle || `Conversation ${link.conversationId}`}
                      {link.label ? ` — ${link.label}` : ""}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))
      )}
    </section>
  );
}

export function GraphDrawer({
  isOpen,
  onClose,
  graph,
  conceptLinks,
  selectedConceptIds,
  onOpenConversation,
  conversationId,
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
          conversationId={conversationId}
          selectedTurnId={selectedTurnId}
          onSelectTurn={onSelectTurn}
          onChanged={onRefresh}
        />
      </div>

      <ConceptLinksPanel
        concepts={conceptLinks.concepts}
        selectedConceptIds={selectedConceptIds}
        onOpenConversation={onOpenConversation}
      />

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
