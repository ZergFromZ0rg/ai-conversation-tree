import { useEffect, useMemo, useState } from "react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from "@xyflow/react";
import { api } from "../api";

const COLUMNS = 4;
const COLUMN_WIDTH = 300;
const NODE_WIDTH = 240;
const NODE_HEIGHT = 60;
const NODE_GAP = 14;
const HEADER_HEIGHT = 34;
const ROW_GAP = 52;

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

function truncate(text, limit = 64) {
  const normalized = (text ?? "").replace(/\s+/g, " ").trim();
  return normalized.length <= limit ? normalized : `${normalized.slice(0, limit - 1)}…`;
}

function ConceptNode({ data }) {
  return (
    <div className="conceptNode" style={{ borderColor: data.color }}>
      <Handle type="target" position={Position.Left} className="conceptNodeHandle" />
      <div className="conceptNodeTitle" style={{ color: data.color }}>
        {data.conversationTitle}
      </div>
      <div className="conceptNodeLabel">{truncate(data.label)}</div>
      <Handle type="source" position={Position.Right} className="conceptNodeHandle" />
    </div>
  );
}

const nodeTypes = { conceptNode: ConceptNode };

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
  let rowTop = 0;
  for (let start = 0; start < conversations.length; start += COLUMNS) {
    const row = conversations.slice(start, start + COLUMNS);
    const tallest = Math.max(...row.map((conversation) => conversation.concepts.length));

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
          flowNodes.push({
            id: `${concept.conversationId}:${concept.conceptId}`,
            position: {
              x,
              y: rowTop + HEADER_HEIGHT + index * (NODE_HEIGHT + NODE_GAP),
            },
            type: "conceptNode",
            data: {
              label: concept.label,
              conversationTitle: conversation.conversationTitle,
              conversationId: concept.conversationId,
              color,
            },
            draggable: false,
            style: { width: NODE_WIDTH },
          });
        });
    });

    rowTop += HEADER_HEIGHT + tallest * (NODE_HEIGHT + NODE_GAP) + ROW_GAP;
  }

  const flowEdges = graph.edges.map((edge, index) => {
    const color = edge.kind === "same" ? "#60a5fa" : "#94a3b8";
    return {
      id: `${edge.a.conversationId}:${edge.a.conceptId}-${edge.b.conversationId}:${edge.b.conceptId}-${index}`,
      source: `${edge.a.conversationId}:${edge.a.conceptId}`,
      target: `${edge.b.conversationId}:${edge.b.conceptId}`,
      type: "bezier",
      label: `${edge.kind} ${edge.score.toFixed(2)}`,
      style: { stroke: color, strokeWidth: edge.kind === "same" ? 2.4 : 1.8, strokeDasharray: edge.kind === "same" ? undefined : "6 4" },
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
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .getConceptGraph()
      .then((payload) => {
        if (!cancelled) setGraph(payload);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const layout = useMemo(() => (graph ? buildLayout(graph) : null), [graph]);
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

        <div className="workspaceMapBody">
          {error ? (
            <div className="graphEmpty">Couldn&rsquo;t load the concept graph.</div>
          ) : !layout ? (
            <div className="graphEmpty">Loading…</div>
          ) : layout.flowNodes.length === 0 ? (
            <div className="graphEmpty">No concepts yet — send some messages first.</div>
          ) : (
            <ReactFlow
              nodes={layout.flowNodes}
              edges={layout.flowEdges}
              nodeTypes={allNodeTypes}
              fitView
              fitViewOptions={{ padding: 0.15 }}
              nodesDraggable={false}
              nodesConnectable={false}
              onNodeClick={(_, node) => {
                if (node.type === "conceptNode") {
                  onOpenConversation(node.data.conversationId);
                  onClose();
                }
              }}
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
