import { useCallback, useMemo, useState } from "react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import { api } from "../api";
import { buildLanePath, edgeTypes as routedEdgeTypes } from "../edgeRouting";

// How far past the widest rank the routing "highway" sits, for edges that
// skip more than one rank (see buildRankRouting below).
const HIGHWAY_MARGIN = 60;

const edgeColors = {
  continuation: "#34d399",
  branch: "#f59e0b",
  related: "#60a5fa",
};

const relationPriority = {
  continuation: 0,
  branch: 1,
  related: 2,
};

const EDGE_LABELS = ["continuation", "branch", "related"];
const MANUAL_EDGE_CONFIDENCE = 1.0;

const NODE_WIDTH = 240;
const NODE_HEIGHT = 92;

function truncatePreview(text, limit = 90) {
  const normalized = (text ?? "").replace(/\s+/g, " ").trim();
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, limit - 1)}…`;
}

function TurnNode({ data }) {
  const className = [
    "turnNode",
    data.isRoot ? "turnNodeRoot" : "",
    data.isSelected ? "turnNodeSelected" : "",
    data.isPending ? "turnNodePending" : "",
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={className}>
      <Handle type="target" position={Position.Top} className="turnNodeHandle" />
      <div className="turnNodeTitle">{data.label}</div>
      <div className="turnNodePreview">{data.preview || "No message text"}</div>
      <Handle type="source" position={Position.Bottom} className="turnNodeHandle" />
    </div>
  );
}

const nodeTypes = { turnNode: TurnNode };

// Pick one incoming edge per node as its "primary" parent: the tightest
// relationship (continuation > branch > related), breaking ties by confidence
// then parent id. Drives both the layout spine and edge styling.
function primaryParentsByNode(nodes, edges) {
  const incomingByTarget = new Map();
  for (const edge of edges) {
    const list = incomingByTarget.get(edge.toTurnId) ?? [];
    list.push(edge);
    incomingByTarget.set(edge.toTurnId, list);
  }

  const primaryByNode = new Map();
  for (const node of nodes) {
    const incoming = (incomingByTarget.get(node.id) ?? []).slice().sort((left, right) => {
      const leftPriority = relationPriority[left.label] ?? 99;
      const rightPriority = relationPriority[right.label] ?? 99;
      if (leftPriority !== rightPriority) return leftPriority - rightPriority;
      if (left.confidence !== right.confidence) return right.confidence - left.confidence;
      return left.fromTurnId - right.fromTurnId;
    });
    if (incoming.length > 0) {
      primaryByNode.set(node.id, incoming[0]);
    }
  }
  return primaryByNode;
}

// The y dagre assigns is identical (within float noise) for every node in
// the same rank when rankdir is TB — round to bucket nodes into ranks and
// get each rank's y band, mirroring WorkspaceMap's row bands but computed
// from the layout instead of a fixed grid.
function buildRankBands(nodeIds, dagreGraph) {
  const yById = new Map(nodeIds.map((id) => [id, Math.round(dagreGraph.node(id)?.y ?? 0)]));
  const rankYs = [...new Set(yById.values())].sort((a, b) => a - b);
  const rankIndexById = new Map([...yById.entries()].map(([id, y]) => [id, rankYs.indexOf(y)]));
  const rankTops = rankYs.map((y) => y - NODE_HEIGHT / 2);
  const rankBottoms = rankYs.map((y) => y + NODE_HEIGHT / 2);
  return { rankIndexById, rankTops, rankBottoms };
}

// A non-primary edge can span ranks in either direction (dagre only lays out
// the primary spine, so a "related"/duplicate edge's two turns can land with
// the semantic target above the semantic source). Route by physical position
// instead of source/target: whichever endpoint is higher up exits from its
// bottom band, whichever is lower enters from its top band — the path is
// still built source-first/target-last so arrowheads stay correct.
function buildRankRouting(sourceId, targetId, positionsById, rankIndexById, rankTops, rankBottoms, highwayX, jitter) {
  const source = positionsById.get(sourceId);
  const target = positionsById.get(targetId);
  const sourceRank = rankIndexById.get(sourceId);
  const targetRank = rankIndexById.get(targetId);
  if (!source || !target || sourceRank === targetRank) {
    return null;
  }

  const laneBelow = (rank) => (rankBottoms[rank] + rankTops[rank + 1]) / 2;
  const laneAbove = (rank) => (rankTops[rank] + rankBottoms[rank - 1]) / 2;

  // positionsById holds dagre's *center* coordinates, so the top/bottom edge
  // of a node is center ± half its height (not ±0 / ±full height).
  const sourceBelowTarget = sourceRank > targetRank;
  const sourceExitY = sourceBelowTarget ? source.y - NODE_HEIGHT / 2 : source.y + NODE_HEIGHT / 2;
  const targetEnterY = sourceBelowTarget ? target.y + NODE_HEIGHT / 2 : target.y - NODE_HEIGHT / 2;
  const sourceLaneY = sourceBelowTarget ? laneAbove(sourceRank) : laneBelow(sourceRank);
  const targetLaneY = sourceBelowTarget ? laneBelow(targetRank) : laneAbove(targetRank);

  return buildLanePath(source.x, sourceExitY, sourceLaneY, target.x, targetEnterY, targetLaneY, highwayX, jitter);
}

function buildFlowLayout(nodes, edges, selectedTurnId, pendingTurnId) {
  const primaryParentByNode = primaryParentsByNode(nodes, edges);

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: "TB", nodesep: 44, ranksep: 84, marginx: 24, marginy: 24 });

  for (const node of nodes) {
    dagreGraph.setNode(String(node.id), { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const [nodeId, edge] of primaryParentByNode.entries()) {
    if (nodes.some((node) => node.id === edge.fromTurnId)) {
      dagreGraph.setEdge(String(edge.fromTurnId), String(nodeId));
    }
  }
  dagre.layout(dagreGraph);

  const nodeIds = nodes.map((node) => String(node.id));
  const { rankIndexById, rankTops, rankBottoms } = buildRankBands(nodeIds, dagreGraph);
  const positionsById = new Map(
    nodeIds.map((id) => {
      const laidOut = dagreGraph.node(id);
      return [id, { x: laidOut?.x ?? 0, y: laidOut?.y ?? 0 }];
    }),
  );
  const highwayX = Math.max(...[...positionsById.values()].map((p) => p.x)) + NODE_WIDTH / 2 + HIGHWAY_MARGIN;

  const flowNodes = nodes.map((node) => {
    const laidOut = dagreGraph.node(String(node.id));
    return {
      id: String(node.id),
      position: {
        x: (laidOut?.x ?? 0) - NODE_WIDTH / 2,
        y: (laidOut?.y ?? 0) - NODE_HEIGHT / 2,
      },
      type: "turnNode",
      data: {
        label: `Turn ${node.id + 1}`,
        preview: truncatePreview(node.userText),
        isRoot: node.root,
        isSelected: node.id === selectedTurnId,
        isPending: node.id === pendingTurnId,
      },
      draggable: false,
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      style: { width: NODE_WIDTH },
    };
  });

  const primaryEdgeKeys = new Set(
    Array.from(primaryParentByNode.entries()).map(
      ([nodeId, edge]) => `${edge.fromTurnId}:${nodeId}:${edge.label}`,
    ),
  );

  const flowEdges = edges.map((edge, index) => {
    const color = edgeColors[edge.label] ?? "#94a3b8";
    const edgeKey = `${edge.fromTurnId}:${edge.toTurnId}:${edge.label}`;
    const isPrimaryEdge = primaryEdgeKeys.has(edgeKey);
    const isBranchLike = edge.label === "branch" || edge.label === "related" || !isPrimaryEdge;
    const isBidirectionalRelated = edge.label === "related";
    const sourceId = String(edge.fromTurnId);
    const targetId = String(edge.toTurnId);
    const routed = isBranchLike
      ? buildRankRouting(sourceId, targetId, positionsById, rankIndexById, rankTops, rankBottoms, highwayX, index)
      : null;
    return {
      id: String(edge.id ?? `${edge.fromTurnId}-${edge.toTurnId}-${edge.label}`),
      source: sourceId,
      target: targetId,
      type: routed ? "routed" : isBranchLike ? "bezier" : "smoothstep",
      label: `${edge.label} ${edge.confidence.toFixed(2)}${edge.origin === "manual" ? " · pinned" : ""}`,
      data: {
        edgeId: edge.id,
        origin: edge.origin,
        fromTurnId: edge.fromTurnId,
        toTurnId: edge.toTurnId,
        label: edge.label,
        path: routed?.d,
        labelX: routed?.labelX,
        labelY: routed?.labelY,
      },
      style: {
        stroke: color,
        strokeWidth: edge.label === "continuation" && isPrimaryEdge ? 2.8 : 2,
        strokeDasharray: edge.label === "related" || !isPrimaryEdge ? "6 4" : undefined,
      },
      labelStyle: { fill: color, fontSize: 11, fontWeight: 600 },
      labelBgStyle: { fill: "var(--graph-label-bg)", fillOpacity: 0.88 },
      markerStart: isBidirectionalRelated ? { type: MarkerType.ArrowClosed, color } : undefined,
      markerEnd: { type: MarkerType.ArrowClosed, color },
      zIndex: isPrimaryEdge ? 2 : 1,
      pathOptions:
        isBranchLike && !routed
          ? { curvature: edge.label === "branch" ? 0.45 : 0.3 }
          : !isBranchLike
            ? { offset: 26, borderRadius: 14 }
            : undefined,
    };
  });

  return { flowNodes, flowEdges };
}

export function TurnGraph({ nodes, edges, conversationId, selectedTurnId, onSelectTurn, onChanged }) {
  const [editMode, setEditMode] = useState(false);
  const [edgeLabel, setEdgeLabel] = useState("related");
  const [pending, setPending] = useState(null); // turnId
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState(null);

  const toggleEditMode = useCallback(() => {
    setEditMode((open) => !open);
    setPending(null);
    setActionError(null);
  }, []);

  const linkTurns = useCallback(
    async (fromTurnId, toTurnId) => {
      setBusy(true);
      setActionError(null);
      try {
        // Re-clicking an already-manually-linked pair relabels it instead of
        // creating a duplicate edge — the same two-click gesture covers both.
        const existingManual = edges.find(
          (edge) => edge.fromTurnId === fromTurnId && edge.toTurnId === toTurnId && edge.origin === "manual",
        );
        if (existingManual) {
          await api.updateTurnEdge(existingManual.id, edgeLabel, MANUAL_EDGE_CONFIDENCE);
        } else {
          await api.createTurnEdge(conversationId, fromTurnId, toTurnId, edgeLabel, MANUAL_EDGE_CONFIDENCE);
        }
        setPending(null);
        await onChanged();
      } catch {
        setActionError("Couldn't save that edge.");
      } finally {
        setBusy(false);
      }
    },
    [conversationId, edgeLabel, edges, onChanged],
  );

  const removeEdge = useCallback(
    async (edgeId) => {
      setBusy(true);
      setActionError(null);
      try {
        await api.deleteTurnEdge(edgeId);
        await onChanged();
      } catch {
        setActionError("Couldn't remove that edge.");
      } finally {
        setBusy(false);
      }
    },
    [onChanged],
  );

  const handleNodeClick = useCallback(
    (_, node) => {
      if (busy) {
        return;
      }
      if (!editMode) {
        onSelectTurn(Number(node.id));
        return;
      }
      const turnId = Number(node.id);
      if (pending === null) {
        setPending(turnId);
        return;
      }
      if (pending === turnId) {
        setPending(null);
        return;
      }
      void linkTurns(Math.min(pending, turnId), Math.max(pending, turnId));
    },
    [editMode, pending, busy, onSelectTurn, linkTurns],
  );

  const handleEdgeClick = useCallback(
    (_, edge) => {
      if (!editMode || busy) {
        return;
      }
      if (edge.data.origin !== "manual") {
        setActionError("Only a manually-created edge can be removed here.");
        return;
      }
      void removeEdge(edge.data.edgeId);
    },
    [editMode, busy, removeEdge],
  );

  const { flowNodes, flowEdges } = useMemo(
    () => buildFlowLayout(nodes, edges, selectedTurnId, pending),
    [nodes, edges, selectedTurnId, pending],
  );

  if (!nodes.length) {
    return <div className="graphEmpty">No turns yet — send a message to build the graph.</div>;
  }

  return (
    <div className="turnGraphRoot">
      <div className="workspaceMapToolbar turnGraphToolbar">
        <button
          type="button"
          className={`ghostButton${editMode ? " ghostButtonActive" : ""}`}
          onClick={toggleEditMode}
        >
          {editMode ? "Done editing" : "Edit edges"}
        </button>
        {editMode ? (
          <div className="linkKindToggle" role="group" aria-label="Edge label">
            {EDGE_LABELS.map((label) => (
              <button
                key={label}
                type="button"
                className={`linkKindOption${edgeLabel === label ? " linkKindOptionActive" : ""}`}
                onClick={() => setEdgeLabel(label)}
              >
                {label}
              </button>
            ))}
          </div>
        ) : null}
        {editMode ? (
          <span className="workspaceMapHint">
            {actionError
              ? actionError
              : pending !== null
                ? "Click another turn to link it, or click it again to cancel."
                : "Click a turn, then another to link or relabel them. Click a pinned edge to remove it."}
          </span>
        ) : null}
      </div>
      <div className="turnGraphCanvas">
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={nodeTypes}
          edgeTypes={routedEdgeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          nodesDraggable={false}
          nodesConnectable={false}
          onNodeClick={handleNodeClick}
          onEdgeClick={handleEdgeClick}
          minZoom={0.2}
          maxZoom={1.75}
          colorMode="system"
        >
          <Background gap={20} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
}
