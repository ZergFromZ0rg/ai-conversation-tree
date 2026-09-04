import { useCallback, useRef, useState } from "react";
import { TurnGraph } from "./TurnGraph";

const legend = [
  { key: "continuation", label: "Continuation" },
  { key: "branch", label: "Branch" },
  { key: "related", label: "Related" },
];

const WIDTH_STORAGE_KEY = "act.drawerWidth";
const MIN_WIDTH = 320;
const MAX_WIDTH = 900;

function readWidthPreference() {
  try {
    const stored = Number(window.localStorage.getItem(WIDTH_STORAGE_KEY));
    return stored >= MIN_WIDTH && stored <= MAX_WIDTH ? stored : 480;
  } catch {
    return 480;
  }
}

function clampWidth(width) {
  const viewportMax = Math.min(MAX_WIDTH, window.innerWidth * 0.7);
  return Math.min(viewportMax, Math.max(MIN_WIDTH, width));
}

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
  const [width, setWidth] = useState(readWidthPreference);
  const [resizing, setResizing] = useState(false);
  const dragRef = useRef(null);

  const handlePointerDown = useCallback(
    (event) => {
      event.preventDefault();
      dragRef.current = { startX: event.clientX, startWidth: width };
      setResizing(true);

      function onPointerMove(moveEvent) {
        const { startX, startWidth } = dragRef.current;
        // The handle sits on the drawer's left edge — dragging left (negative
        // delta) widens the drawer, dragging right narrows it.
        const next = clampWidth(startWidth + (startX - moveEvent.clientX));
        dragRef.current.latestWidth = next;
        setWidth(next);
      }

      function onPointerUp() {
        setResizing(false);
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
        try {
          window.localStorage.setItem(WIDTH_STORAGE_KEY, String(dragRef.current.latestWidth ?? width));
        } catch {
          /* storage unavailable — ignore */
        }
      }

      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
    },
    [width],
  );

  return (
    <section
      className={`graphDrawer${isOpen ? " graphDrawerOpen" : ""}`}
      aria-hidden={!isOpen}
      style={{ width }}
    >
      <div
        className={`graphDrawerResizeHandle${resizing ? " graphDrawerResizeHandle--active" : ""}`}
        onPointerDown={handlePointerDown}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize conversation graph panel"
      />
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
