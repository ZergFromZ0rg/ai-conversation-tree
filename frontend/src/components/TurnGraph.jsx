import { useCallback, useMemo, useState } from "react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import { api } from "../api";

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

  const flowEdges = edges.map((edge) => {
    const color = edgeColors[edge.label] ?? "#94a3b8";
    const edgeKey = `${edge.fromTurnId}:${edge.toTurnId}:${edge.label}`;
    const isPrimaryEdge = primaryEdgeKeys.has(edgeKey);
    const isBranchLike = edge.label === "branch" || edge.label === "related" || !isPrimaryEdge;
    const isBidirectionalRelated = edge.label === "related";
    return {
      id: String(edge.id ?? `${edge.fromTurnId}-${edge.toTurnId}-${edge.label}`),
      source: String(edge.fromTurnId),
      target: String(edge.toTurnId),
      type: isBranchLike ? "bezier" : "smoothstep",
      label: `${edge.label} ${edge.confidence.toFixed(2)}${edge.origin === "manual" ? " · pinned" : ""}`,
      data: {
        edgeId: edge.id,
        origin: edge.origin,
        fromTurnId: edge.fromTurnId,
        toTurnId: edge.toTurnId,
        label: edge.label,
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
      pathOptions: isBranchLike
        ? { curvature: edge.label === "branch" ? 0.45 : 0.3 }
        : { offset: 26, borderRadius: 14 },
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
