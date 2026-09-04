import { useCallback, useEffect, useMemo, useState } from "react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from "@xyflow/react";
import { api } from "../api";
import { buildLanePath, edgeTypes } from "../edgeRouting";

const COLUMNS = 4;
const COLUMN_WIDTH = 300;
const NODE_WIDTH = 240;
const NODE_HEIGHT = 60;
const NODE_GAP = 14;
const HEADER_HEIGHT = 34;
const ROW_GAP = 52;
// Fixed vertical "highway" x, past every column in every row, that an edge
// between rows more than one apart is routed through — see buildRoutedPath.
const HIGHWAY_X = COLUMNS * COLUMN_WIDTH + 60;

const LINK_KINDS = ["related", "same"];

// Distinct-ish hues cycled per conversation so a cluster reads as one colour.
const conversationColors = [
  "#60a5fa",
  "#34d399",
  "#f59e0b",
  "#c084fc",
  "#f472b6",
  "#22d3ee",
  "#a3e635",
  "#fb7185",
];

// A pinned (manual) link's colour — deliberately outside conversationColors
// so it never coincides with a node's own hue.
const manualLinkColor = "#fbbf24";

function truncate(text, limit = 64) {
  const normalized = (text ?? "").replace(/\s+/g, " ").trim();
  return normalized.length <= limit ? normalized : `${normalized.slice(0, limit - 1)}…`;
}

function ConceptNode({ data }) {
  const className = ["conceptNode", data.isPending ? "conceptNodePending" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={className} style={{ borderColor: data.isPending ? undefined : data.color }}>
      {/* Edges route via own explicit paths (buildRoutedPath), entering/leaving
          top or bottom depending which is closer to their row's free lane —
          these handles exist for React Flow's bookkeeping, not for geometry. */}
      <Handle type="target" position={Position.Top} className="conceptNodeHandle" />
      <div className="conceptNodeTitle" style={{ color: data.color }}>
        {data.conversationTitle}
      </div>
      <div className="conceptNodeLabel">{truncate(data.label)}</div>
      <Handle type="source" position={Position.Bottom} className="conceptNodeHandle" />
    </div>
  );
}

const nodeTypes = { conceptNode: ConceptNode };

// Each row of clusters sits in a "band" of node cards with a genuinely empty
// horizontal gap (ROW_GAP) above and below it — nothing is ever drawn there.
// ownLaneY picks the gap adjacent to a given row: below it, or above it for
// the last row (which has no gap below). Returns null only when there is
// exactly one row total, so no gap exists anywhere to route through.
function ownLaneY(rowIndex, rowTops, rowBandBottoms) {
  const numRows = rowTops.length;
  if (rowIndex < numRows - 1) {
    return { y: (rowBandBottoms[rowIndex] + rowTops[rowIndex + 1]) / 2, side: "bottom" };
  }
  if (rowIndex > 0) {
    return { y: (rowBandBottoms[rowIndex - 1] + rowTops[rowIndex]) / 2, side: "top" };
  }
  return null;
}

// Routes an edge entirely through the empty row-gap lanes (and, when the two
// concepts' lanes differ, a fixed vertical "highway" past every column) so it
// never crosses a node card — unlike a direct line, which cuts straight
// through whatever sits between two far-apart clusters. Returns null for
// edges a straight line already handles fine (same conversation; same row,
// adjacent columns) or when no lane exists to route through (single-row
// workspace) — callers fall back to a plain bezier in both cases.
function buildRoutedPath(sourceId, targetId, positionsById, metaById, rowTops, rowBandBottoms, laneIndex) {
  const source = positionsById.get(sourceId);
  const target = positionsById.get(targetId);
  const sourceMeta = metaById.get(sourceId);
  const targetMeta = metaById.get(targetId);
  if (!source || !target || !sourceMeta || !targetMeta) {
    return null;
  }
  if (sourceMeta.conversationId === targetMeta.conversationId) {
    return null;
  }
  if (sourceMeta.row === targetMeta.row && Math.abs(sourceMeta.col - targetMeta.col) <= 1) {
    return null;
  }

  const sourceLane = ownLaneY(sourceMeta.row, rowTops, rowBandBottoms);
  const targetLane = ownLaneY(targetMeta.row, rowTops, rowBandBottoms);
  if (!sourceLane || !targetLane) {
    return null;
  }

  // Nudge parallel edges apart so two links sharing a lane don't overlap.
  const jitter = ((laneIndex % 3) - 1) * 6;
  const sourceCenterX = source.x + NODE_WIDTH / 2;
  const targetCenterX = target.x + NODE_WIDTH / 2;
  const sourceExitY = sourceLane.side === "bottom" ? source.y + NODE_HEIGHT : source.y;
  const targetExitY = targetLane.side === "bottom" ? target.y + NODE_HEIGHT : target.y;

  return buildLanePath(
    sourceCenterX,
    sourceExitY,
    sourceLane.y,
    targetCenterX,
    targetExitY,
    targetLane.y,
    HIGHWAY_X,
    jitter,
  );
}

function buildLayout(graph) {
  const byConversation = new Map();
  for (const node of graph.nodes) {
    const bucket = byConversation.get(node.conversationId) ?? {
      conversationId: node.conversationId,
      conversationTitle: node.conversationTitle || `Conversation ${node.conversationId}`,
      concepts: [],
    };
    bucket.concepts.push(node);
    byConversation.set(node.conversationId, bucket);
  }

  const conversations = [...byConversation.values()].sort(
    (left, right) => left.conversationId - right.conversationId,
  );

  const colorByConversation = new Map();
  conversations.forEach((conversation, index) => {
    colorByConversation.set(
      conversation.conversationId,
      conversationColors[index % conversationColors.length],
    );
  });

  const flowNodes = [];
  const positionsById = new Map();
  const metaById = new Map();
  const rowTops = [];
  const rowBandBottoms = [];
  let rowTop = 0;
  let rowIndex = 0;
  for (let start = 0; start < conversations.length; start += COLUMNS) {
    const row = conversations.slice(start, start + COLUMNS);
    const tallest = Math.max(...row.map((conversation) => conversation.concepts.length));
    rowTops.push(rowTop);

    row.forEach((conversation, column) => {
      const x = column * COLUMN_WIDTH;
      const color = colorByConversation.get(conversation.conversationId);

      flowNodes.push({
        id: `title:${conversation.conversationId}`,
        position: { x, y: rowTop },
        data: { label: conversation.conversationTitle, color },
        type: "conceptGroupTitle",
        draggable: false,
        selectable: false,
        style: { width: NODE_WIDTH },
      });

      conversation.concepts
        .slice()
        .sort((left, right) => left.conceptId - right.conceptId)
        .forEach((concept, index) => {
          const id = `${concept.conversationId}:${concept.conceptId}`;
          const position = { x, y: rowTop + HEADER_HEIGHT + index * (NODE_HEIGHT + NODE_GAP) };
          flowNodes.push({
            id,
            position,
            type: "conceptNode",
            data: {
              label: concept.label,
              conversationTitle: conversation.conversationTitle,
              conversationId: concept.conversationId,
              conceptKey: concept.conceptKey,
              color,
            },
            draggable: false,
            style: { width: NODE_WIDTH },
          });
          positionsById.set(id, position);
          metaById.set(id, { row: rowIndex, col: column, conversationId: concept.conversationId });
        });
    });

    rowBandBottoms.push(rowTop + HEADER_HEIGHT + tallest * (NODE_HEIGHT + NODE_GAP) - NODE_GAP);
    rowTop += HEADER_HEIGHT + tallest * (NODE_HEIGHT + NODE_GAP) + ROW_GAP;
    rowIndex += 1;
  }

  const flowEdges = graph.edges.map((edge, index) => {
    const isManual = edge.origin === "manual";
    // manualLinkColor is deliberately not in conversationColors — a pinned
    // link must never coincide with whichever hue a node happens to have.
    const color = isManual ? manualLinkColor : edge.kind === "same" ? "#60a5fa" : "#94a3b8";
    const sourceId = `${edge.a.conversationId}:${edge.a.conceptId}`;
    const targetId = `${edge.b.conversationId}:${edge.b.conceptId}`;
    const routed = buildRoutedPath(sourceId, targetId, positionsById, metaById, rowTops, rowBandBottoms, index);
    const label = isManual ? `${edge.kind} · pinned` : `${edge.kind} ${edge.score.toFixed(2)}`;

    return {
      id: String(edge.id),
      source: sourceId,
      target: targetId,
      type: routed ? "routed" : "bezier",
      label,
      data: { linkId: edge.id, origin: edge.origin, path: routed?.d, labelX: routed?.labelX, labelY: routed?.labelY },
      style: {
        stroke: color,
        strokeWidth: isManual || edge.kind === "same" ? 2.4 : 1.8,
        strokeDasharray: isManual ? undefined : edge.kind === "same" ? undefined : "6 4",
      },
      labelStyle: { fill: color, fontSize: 11, fontWeight: 600 },
      labelBgStyle: { fill: "var(--graph-label-bg)", fillOpacity: 0.88 },
      markerStart: { type: MarkerType.ArrowClosed, color },
      markerEnd: { type: MarkerType.ArrowClosed, color },
    };
  });

  return { flowNodes, flowEdges };
}

function GroupTitleNode({ data }) {
  return <div className="conceptGroupTitle" style={{ color: data.color }}>{data.label}</div>;
}

const allNodeTypes = { ...nodeTypes, conceptGroupTitle: GroupTitleNode };

export function WorkspaceMap({ onClose, onOpenConversation }) {
  const [graph, setGraph] = useState(null);
  const [loadError, setLoadError] = useState(false);
  const [linkMode, setLinkMode] = useState(false);
  const [linkKind, setLinkKind] = useState("related");
  const [pending, setPending] = useState(null); // { nodeId, conceptKey }
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState(null);

  const reloadGraph = useCallback(async () => {
    try {
      const payload = await api.getConceptGraph();
      setGraph(payload);
      setLoadError(false);
    } catch {
      setLoadError(true);
    }
  }, []);

  useEffect(() => {
    void reloadGraph();
  }, [reloadGraph]);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const toggleLinkMode = useCallback(() => {
    setLinkMode((open) => !open);
    setPending(null);
    setActionError(null);
  }, []);

  const createLink = useCallback(
    async (keyA, keyB) => {
      setBusy(true);
      setActionError(null);
      try {
        await api.createConceptLink(keyA, keyB, linkKind);
        setPending(null);
        await reloadGraph();
      } catch {
        setActionError("Couldn't create that link.");
      } finally {
        setBusy(false);
      }
    },
    [linkKind, reloadGraph],
  );

  const removeLink = useCallback(
    async (linkId) => {
      setBusy(true);
      setActionError(null);
      try {
        await api.deleteConceptLink(linkId);
        await reloadGraph();
      } catch {
        setActionError("Couldn't remove that link.");
      } finally {
        setBusy(false);
      }
    },
    [reloadGraph],
  );

  const handleNodeClick = useCallback(
    (_, node) => {
      if (node.type !== "conceptNode" || busy) {
        return;
      }
      if (!linkMode) {
        onOpenConversation(node.data.conversationId);
        onClose();
        return;
      }
      if (!pending) {
        setPending({ nodeId: node.id, conceptKey: node.data.conceptKey });
        return;
      }
      if (pending.nodeId === node.id) {
        setPending(null);
        return;
      }
      void createLink(pending.conceptKey, node.data.conceptKey);
    },
    [linkMode, pending, busy, createLink, onOpenConversation, onClose],
  );

  const handleEdgeClick = useCallback(
    (_, edge) => {
      if (!linkMode || busy) {
        return;
      }
      if (edge.data.origin !== "manual") {
        setActionError("Only a link you pinned by hand can be removed here.");
        return;
      }
      void removeLink(edge.data.linkId);
    },
    [linkMode, busy, removeLink],
  );

  const layout = useMemo(() => (graph ? buildLayout(graph) : null), [graph]);
  const nodesForRender = useMemo(() => {
    if (!layout) return [];
    if (!pending) return layout.flowNodes;
    return layout.flowNodes.map((node) =>
      node.id === pending.nodeId ? { ...node, data: { ...node.data, isPending: true } } : node,
    );
  }, [layout, pending]);
  const edgeCount = graph?.edges.length ?? 0;

  return (
    <div className="workspaceMapBackdrop" onClick={onClose}>
      <div className="workspaceMap" onClick={(event) => event.stopPropagation()}>
        <header className="workspaceMapHeader">
          <div className="graphDrawerTitle">
            <span>Workspace map</span>
            <span className="graphDrawerMeta">
              {graph ? `${graph.nodes.length} concepts · ${edgeCount} links` : "…"}
            </span>
          </div>
          <button type="button" className="iconButton" aria-label="Close map" onClick={onClose}>
            ×
          </button>
        </header>

        <div className="workspaceMapToolbar">
          <button
            type="button"
            className={`ghostButton${linkMode ? " ghostButtonActive" : ""}`}
            onClick={toggleLinkMode}
          >
            {linkMode ? "Done linking" : "Link concepts"}
          </button>
          {linkMode ? (
            <div className="linkKindToggle" role="group" aria-label="Link kind">
              {LINK_KINDS.map((kind) => (
                <button
                  key={kind}
                  type="button"
                  className={`linkKindOption${linkKind === kind ? " linkKindOptionActive" : ""}`}
                  onClick={() => setLinkKind(kind)}
                >
                  {kind}
                </button>
              ))}
            </div>
          ) : null}
          <span className="workspaceMapHint">
            {actionError
              ? actionError
              : !linkMode
                ? "Click a concept to jump to that conversation."
                : pending
                  ? "Click another concept to link it, or click it again to cancel."
                  : "Click a concept, then another to link them. Click a pinned link to remove it."}
          </span>
        </div>

        <div className="workspaceMapBody">
          {loadError ? (
            <div className="graphEmpty">Couldn&rsquo;t load the concept graph.</div>
          ) : !layout ? (
            <div className="graphEmpty">Loading…</div>
          ) : layout.flowNodes.length === 0 ? (
            <div className="graphEmpty">No concepts yet — send some messages first.</div>
          ) : (
            <ReactFlow
              nodes={nodesForRender}
              edges={layout.flowEdges}
              nodeTypes={allNodeTypes}
              edgeTypes={edgeTypes}
              fitView
              fitViewOptions={{ padding: 0.15 }}
              nodesDraggable={false}
              nodesConnectable={false}
              onNodeClick={handleNodeClick}
              onEdgeClick={handleEdgeClick}
              minZoom={0.1}
              maxZoom={1.75}
              colorMode="system"
            >
              <Background gap={20} size={1} />
              <Controls showInteractive={false} />
            </ReactFlow>
          )}
        </div>
      </div>
    </div>
  );
}
