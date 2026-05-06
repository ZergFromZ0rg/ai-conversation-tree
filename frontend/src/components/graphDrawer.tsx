import { memo, useMemo } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import type { GraphEdge, GraphNode } from "../types";

type GraphDrawerProps = {
  isOpen: boolean;
  nodes: GraphNode[];
  edges: GraphEdge[];
  onClose: () => void;
  onRefresh: () => void;
  onReclassify: () => void;
};

type TurnNodeData = {
  turnId: number;
  userText: string;
  aiText: string;
  conceptIds: number[];
  root: boolean;
};

const edgeColors: Record<string, string> = {
  continuation: "#34d399",
  branch: "#f59e0b",
  related: "#60a5fa",
};

const columnWidth = 320;
const rowHeight = 170;

const TurnNode = memo(({ data }: { data: TurnNodeData }) => {
  return (
    <div className={`graphTurnNode ${data.root ? "root" : ""}`}>
      <Handle type="target" position={Position.Top} className="graphHandle" />
      <div className="graphNodeHeader">
        <span>Turn {data.turnId}</span>
        <span className="graphNodeConcepts">C{data.conceptIds.join(", ") || "–"}</span>
      </div>
      <div className="graphNodeSection">
        <div className="graphNodeLabel">User</div>
        <div className="graphNodeText">{data.userText}</div>
      </div>
      <div className="graphNodeSection">
        <div className="graphNodeLabel">Assistant</div>
        <div className="graphNodeText graphNodeMuted">{data.aiText}</div>
      </div>
      <Handle type="source" position={Position.Bottom} className="graphHandle" />
    </div>
  );
});

TurnNode.displayName = "TurnNode";

function buildGraphLayout(nodes: GraphNode[], semanticEdges: GraphEdge[]): { flowNodes: Node<TurnNodeData>[]; flowEdges: Edge[] } {
  const turnMap = new Map<number, GraphNode>(nodes.map((node) => [node.id, node]));
  const childrenByParent = new Map<number | null, number[]>();

  for (const node of nodes) {
    const parentId = node.timelineParent;
    const siblings = childrenByParent.get(parentId) ?? [];
    siblings.push(node.id);
    childrenByParent.set(parentId, siblings);
  }

  for (const siblings of childrenByParent.values()) {
    siblings.sort((left, right) => left - right);
  }

  const positions = new Map<number, { x: number; y: number }>();
  let nextLeafX = 0;

  function placeNode(nodeId: number, depth: number) {
    const children = childrenByParent.get(nodeId) ?? [];
    if (!children.length) {
      positions.set(nodeId, { x: nextLeafX * columnWidth, y: depth * rowHeight });
      nextLeafX += 1;
      return;
    }

    for (const childId of children) {
      placeNode(childId, depth + 1);
    }

    const childPositions = children
      .map((childId) => positions.get(childId))
      .filter((position): position is { x: number; y: number } => position !== undefined);
    const averageX =
      childPositions.reduce((total, position) => total + position.x, 0) / Math.max(childPositions.length, 1);
    positions.set(nodeId, { x: averageX, y: depth * rowHeight });
  }

  const rootIds = [...(childrenByParent.get(null) ?? [])];
  for (const rootId of rootIds) {
    placeNode(rootId, 0);
  }

  const flowNodes: Node<TurnNodeData>[] = nodes.map((node) => ({
    id: String(node.id),
    type: "turnNode",
    position: positions.get(node.id) ?? { x: 0, y: 0 },
    data: {
      turnId: node.id,
      userText: node.userText,
      aiText: node.aiText,
      conceptIds: node.conceptIds,
      root: node.root,
    },
    draggable: false,
    selectable: true,
  }));

  const timelineEdges: Edge[] = nodes
    .filter((node) => node.timelineParent !== null)
    .map((node) => ({
      id: `timeline-${node.timelineParent}-${node.id}`,
      source: String(node.timelineParent),
      target: String(node.id),
      type: "smoothstep",
      animated: false,
      label: "timeline",
      style: { stroke: "#4b5563", strokeWidth: 1.5 },
      labelStyle: { fill: "#94a3b8", fontSize: 11 },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#4b5563" },
      zIndex: 0,
    }));

  const flowEdges: Edge[] = [
    ...timelineEdges,
    ...semanticEdges.map((edge) => {
      const color = edgeColors[edge.type] ?? "#60a5fa";
      const timelineTargetParent = turnMap.get(edge.to)?.timelineParent;
      const isTimelineDuplicate = timelineTargetParent === edge.from;

      return {
        id: `semantic-${edge.from}-${edge.to}-${edge.type}`,
        source: String(edge.from),
        target: String(edge.to),
        type: "smoothstep",
        label: `${edge.type} ${edge.confidence.toFixed(2)}`,
        style: {
          stroke: color,
          strokeWidth: isTimelineDuplicate ? 3 : 2,
          strokeDasharray: isTimelineDuplicate ? undefined : "6 6",
        },
        labelStyle: { fill: color, fontSize: 11, fontWeight: 600 },
        markerEnd: { type: MarkerType.ArrowClosed, color },
        zIndex: isTimelineDuplicate ? 2 : 1,
      };
    }),
  ];

  return { flowNodes, flowEdges };
}

export function GraphDrawer({
  isOpen,
  nodes,
  edges,
  onClose,
  onRefresh,
  onReclassify
}: GraphDrawerProps) {
  const { flowNodes, flowEdges } = useMemo(() => buildGraphLayout(nodes, edges), [nodes, edges]);

  return (
    <aside className={`drawer rightDrawer ${isOpen ? "open" : ""}`}>
      <div className="drawerHeader">
        <div className="drawerTitle">Conversation Graph</div>
        <button className="iconButton" onClick={onClose}>✕</button>
      </div>
      <div className="drawerBody">
        <div className="graphMeta">Semantic edges for the active conversation.</div>
        <div className="graphActions">
          <button onClick={onRefresh}>Refresh</button>
          <button onClick={onReclassify}>Reclassify</button>
        </div>
        <div className="graphContainer">
          {isOpen ? (
            <ReactFlow
              nodes={flowNodes}
              edges={flowEdges}
              nodeTypes={{ turnNode: TurnNode }}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable
              minZoom={0.25}
              maxZoom={1.5}
              colorMode="dark"
            >
              <Background color="#334155" gap={20} size={1} />
              <Controls showInteractive={false} />
            </ReactFlow>
          ) : null}
        </div>
      </div>
    </aside>
  );
}
