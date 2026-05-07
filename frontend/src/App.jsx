import { useEffect, useMemo, useState } from "react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow } from "@xyflow/react";

const edgeColors = {
  continuation: "#34d399",
  branch: "#f59e0b",
  related: "#60a5fa",
};

function truncatePreview(text, limit = 90) {
  const normalized = (text ?? "").replace(/\s+/g, " ").trim();
  if (normalized.length <= limit) {
    return normalized;
  }
  return `${normalized.slice(0, limit - 1)}…`;
}

function TurnNode({ data }) {
  return (
    <div className={`turnNode${data.isRoot ? " turnNodeRoot" : ""}`}>
      <Handle type="target" position={Position.Top} className="turnNodeHandle" />
      <div className="turnNodeTitle">{data.label}</div>
      <div className="turnNodePreview">{data.preview || "No message text"}</div>
      <Handle type="source" position={Position.Bottom} className="turnNodeHandle" />
    </div>
  );
}

const nodeTypes = {
  turnNode: TurnNode,
};

function buildFlowLayout(nodes, edges) {
  const edgesByTarget = new Map();
  const childrenByParent = new Map();

  const relationPriority = {
    continuation: 0,
    branch: 1,
    related: 2,
  };

  for (const edge of edges) {
    const existing = edgesByTarget.get(edge.toTurnId) ?? [];
    existing.push(edge);
    edgesByTarget.set(edge.toTurnId, existing);
  }

  const primaryParentByNode = new Map();
  for (const node of nodes) {
    const incomingEdges = (edgesByTarget.get(node.id) ?? []).slice().sort((left, right) => {
      const leftPriority = relationPriority[left.label] ?? 99;
      const rightPriority = relationPriority[right.label] ?? 99;
      if (leftPriority !== rightPriority) {
        return leftPriority - rightPriority;
      }
      if (left.confidence !== right.confidence) {
        return right.confidence - left.confidence;
      }
      return left.fromTurnId - right.fromTurnId;
    });

    if (incomingEdges.length > 0) {
      const primaryParent = incomingEdges[0];
      primaryParentByNode.set(node.id, primaryParent);
      const children = childrenByParent.get(primaryParent.fromTurnId) ?? [];
      children.push({
        childId: node.id,
        relation: primaryParent.label,
        confidence: primaryParent.confidence,
      });
      childrenByParent.set(primaryParent.fromTurnId, children);
    }
  }

  for (const children of childrenByParent.values()) {
    children.sort((left, right) => {
      const leftPriority = relationPriority[left.relation] ?? 99;
      const rightPriority = relationPriority[right.relation] ?? 99;
      if (leftPriority !== rightPriority) {
        return leftPriority - rightPriority;
      }
      if (left.confidence !== right.confidence) {
        return right.confidence - left.confidence;
      }
      return left.childId - right.childId;
    });
  }

  const rowHeight = 190;
  const columnWidth = 300;
  const sideGapUnits = 0.9;
  const rootGapUnits = 1.4;
  const positions = new Map();
  const extentsByNode = new Map();

  function splitChildren(children) {
    const continuationChildren = children.filter((child) => child.relation === "continuation");
    const sideChildren = children.filter((child) => child.relation !== "continuation");
    const leftChildren = [];
    const rightChildren = [];

    sideChildren.forEach((child, index) => {
      if (index % 2 === 0) {
        leftChildren.push(child);
      } else {
        rightChildren.push(child);
      }
    });

    continuationChildren.slice(1).forEach((child, index) => {
      if (index % 2 === 0) {
        rightChildren.push(child);
      } else {
        leftChildren.push(child);
      }
    });

    return {
      trunkChild: continuationChildren[0] ?? null,
      leftChildren,
      rightChildren,
    };
  }

  function computeExtents(nodeId) {
    if (extentsByNode.has(nodeId)) {
      return extentsByNode.get(nodeId);
    }

    const children = childrenByParent.get(nodeId) ?? [];
    const { trunkChild, leftChildren, rightChildren } = splitChildren(children);
    let leftExtent = 0.55;
    let rightExtent = 0.55;

    if (trunkChild) {
      const trunkExtents = computeExtents(trunkChild.childId);
      leftExtent = Math.max(leftExtent, trunkExtents.left);
      rightExtent = Math.max(rightExtent, trunkExtents.right);
    }

    let leftCursor = -sideGapUnits;
    for (const child of leftChildren) {
      const childExtents = computeExtents(child.childId);
      const childCenter = leftCursor - childExtents.right;
      leftExtent = Math.max(leftExtent, -childCenter + childExtents.left);
      rightExtent = Math.max(rightExtent, childCenter + childExtents.right);
      leftCursor = childCenter - childExtents.left - sideGapUnits;
    }

    let rightCursor = sideGapUnits;
    for (const child of rightChildren) {
      const childExtents = computeExtents(child.childId);
      const childCenter = rightCursor + childExtents.left;
      leftExtent = Math.max(leftExtent, -childCenter + childExtents.left);
      rightExtent = Math.max(rightExtent, childCenter + childExtents.right);
      rightCursor = childCenter + childExtents.right + sideGapUnits;
    }

    const extents = { left: leftExtent, right: rightExtent };
    extentsByNode.set(nodeId, extents);
    return extents;
  }

  function placeNode(nodeId, depth, centerXUnits) {
    if (positions.has(nodeId)) {
      return;
    }

    positions.set(nodeId, { x: centerXUnits * columnWidth, y: depth * rowHeight });

    const children = childrenByParent.get(nodeId) ?? [];
    if (!children.length) {
      return;
    }

    const { trunkChild, leftChildren, rightChildren } = splitChildren(children);

    if (trunkChild) {
      placeNode(trunkChild.childId, depth + 1, centerXUnits);
    }

    let leftCursor = centerXUnits - sideGapUnits;
    for (const child of leftChildren) {
      const childExtents = computeExtents(child.childId);
      const childCenter = leftCursor - childExtents.right;
      placeNode(child.childId, depth + 1, childCenter);
      leftCursor = childCenter - childExtents.left - sideGapUnits;
    }

    let rightCursor = centerXUnits + sideGapUnits;
    for (const child of rightChildren) {
      const childExtents = computeExtents(child.childId);
      const childCenter = rightCursor + childExtents.left;
      placeNode(child.childId, depth + 1, childCenter);
      rightCursor = childCenter + childExtents.right + sideGapUnits;
    }
  }

  const rootNodes = nodes
    .filter((node) => !primaryParentByNode.has(node.id))
    .sort((left, right) => left.id - right.id);

  rootNodes.forEach((rootNode, index) => {
    const rootExtents = computeExtents(rootNode.id);
    if (index === 0) {
      placeNode(rootNode.id, 0, rootExtents.left);
      return;
    }

    const previousRoot = rootNodes[index - 1];
    const previousRootExtents = computeExtents(previousRoot.id);
    const previousRootCenter = positions.get(previousRoot.id)?.x ?? 0;
    const previousRootCenterUnits = previousRootCenter / columnWidth;
    const nextCenter = previousRootCenterUnits + previousRootExtents.right + rootGapUnits + rootExtents.left;
    placeNode(rootNode.id, 0, nextCenter);
  });

  const primaryEdgeKeys = new Set(
    Array.from(primaryParentByNode.entries()).map(([nodeId, edge]) => `${edge.fromTurnId}:${nodeId}:${edge.label}`)
  );

  const flowNodes = nodes.map((node) => ({
    id: String(node.id),
    position: positions.get(node.id) ?? { x: 0, y: 0 },
    type: "turnNode",
    data: {
      label: `Turn ${node.id + 1}`,
      preview: truncatePreview(node.userText),
      isRoot: node.root,
    },
    draggable: false,
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
    style: {
      width: 250,
    },
  }));

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
      label: `${edge.label} ${edge.confidence.toFixed(2)}`,
      style: {
        stroke: color,
        strokeWidth: edge.label === "continuation" && isPrimaryEdge ? 2.8 : 2,
        strokeDasharray: edge.label === "related" || !isPrimaryEdge ? "6 4" : undefined,
      },
      labelStyle: { fill: color, fontSize: 11, fontWeight: 600 },
      labelBgStyle: { fill: "#0f1419", fillOpacity: 0.88 },
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

async function apiRequest(path, init) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

export function App() {
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [turns, setTurns] = useState([]);
  const [graph, setGraph] = useState({ nodes: [], edges: [] });
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [status, setStatus] = useState("Loading conversations...");
  const [newConversationTitle, setNewConversationTitle] = useState("");
  const [messageInput, setMessageInput] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function loadConversations() {
    const payload = await apiRequest("/conversations");
    setConversations(payload.conversations);
    if (!conversationId && payload.conversations.length > 0) {
      setConversationId(payload.conversations[0].id);
    }
    if (!payload.conversations.length) {
      setStatus("No conversations yet. Create one through the API first.");
    }
  }

  async function createConversation() {
    setIsSubmitting(true);
    try {
      const payload = await apiRequest("/conversations", {
        method: "POST",
        body: JSON.stringify({
          title: newConversationTitle.trim() || "Local Ollama Test",
        }),
      });
      setNewConversationTitle("");
      await loadConversations();
      setConversationId(payload.conversationId);
      setStatus(`Created conversation ${payload.conversationId}`);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function loadConversationData(activeConversationId) {
    if (!activeConversationId) {
      return;
    }

    setStatus(`Loading conversation ${activeConversationId}...`);
    const [turnsPayload, graphPayload] = await Promise.all([
      apiRequest(`/conversations/${activeConversationId}/turns`),
      apiRequest(`/conversations/${activeConversationId}/graph`),
    ]);
    setTurns(turnsPayload.turns);
    setGraph({ nodes: graphPayload.nodes, edges: graphPayload.edges });
    setSelectedNodeId(graphPayload.nodes[0]?.id ?? null);
    setStatus(`Loaded conversation ${activeConversationId}`);
  }

  async function analyzeConversation() {
    if (!conversationId) {
      return;
    }
    setStatus(`Analyzing conversation ${conversationId}...`);
    await apiRequest(`/conversations/${conversationId}/analyze`, { method: "POST" });
    await loadConversationData(conversationId);
    setStatus(`Analyzed conversation ${conversationId}`);
  }

  async function sendTurn() {
    const userText = messageInput.trim();
    if (!conversationId || !userText) {
      return;
    }

    setIsSubmitting(true);
    setStatus(`Sending turn to conversation ${conversationId}...`);
    try {
      await apiRequest(`/conversations/${conversationId}/turns`, {
        method: "POST",
        body: JSON.stringify({ userText }),
      });
      setMessageInput("");
      await loadConversationData(conversationId);
      await loadConversations();
      setStatus(`Added turn to conversation ${conversationId}`);
    } finally {
      setIsSubmitting(false);
    }
  }

  useEffect(() => {
    void loadConversations();
  }, []);

  useEffect(() => {
    if (conversationId) {
      void loadConversationData(conversationId);
    }
  }, [conversationId]);

  const { flowNodes, flowEdges } = useMemo(() => buildFlowLayout(graph.nodes, graph.edges), [graph]);

  const selectedTurn = turns.find((turn) => turn.id === selectedNodeId) ?? null;

  return (
    <div className="appShell">
      <aside className="sidebar">
        <h1>Graph Viewer</h1>
        <div className="panelLabel">Conversation</div>
        <select
          className="selectInput"
          value={conversationId ?? ""}
          onChange={(event) => setConversationId(Number(event.target.value))}
        >
          {!conversations.length ? <option value="">No conversations</option> : null}
          {conversations.map((conversation) => (
            <option key={conversation.id} value={conversation.id}>
              {conversation.title || `Conversation ${conversation.id}`}
            </option>
          ))}
        </select>

        <div className="panelLabel">New Conversation</div>
        <input
          className="textInput"
          value={newConversationTitle}
          onChange={(event) => setNewConversationTitle(event.target.value)}
          placeholder="Conversation title"
        />
        <div className="buttonRow">
          <button onClick={createConversation} disabled={isSubmitting}>Create</button>
        </div>

        <div className="panelLabel">Send Message</div>
        <textarea
          className="textAreaInput"
          value={messageInput}
          onChange={(event) => setMessageInput(event.target.value)}
          placeholder="Send a user message to Ollama"
          rows={4}
        />
        <div className="buttonRow">
          <button onClick={sendTurn} disabled={isSubmitting || !conversationId || !messageInput.trim()}>
            Send
          </button>
        </div>

        <div className="buttonRow">
          <button onClick={() => conversationId && loadConversationData(conversationId)} disabled={isSubmitting}>Refresh</button>
          <button onClick={analyzeConversation} disabled={isSubmitting || !conversationId}>Analyze</button>
        </div>

        <div className="statusText">{status}</div>

        <div className="panelLabel">Message</div>
        {selectedTurn ? (
          <div className="detailCard">
            <div className="detailHeader">Turn {selectedTurn.id + 1}</div>
            <div className="detailSection">
              <div className="detailLabel">User</div>
              <div className="detailText">{selectedTurn.userText}</div>
            </div>
            <div className="detailSection">
              <div className="detailLabel">Assistant</div>
              <div className="detailText">{selectedTurn.aiText}</div>
            </div>
            <div className="detailSection">
              <div className="detailLabel">Metadata</div>
              <div className="detailText">
                Root: {String(selectedTurn.root)}
                <br />
                Timeline parent: {selectedTurn.timelineParent ?? "None"}
                <br />
                Concepts: {selectedTurn.conceptIds.join(", ") || "None"}
              </div>
            </div>
          </div>
        ) : (
          <div className="emptyDetail">Click a node to inspect the message.</div>
        )}
      </aside>

      <main className="graphPanel">
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          nodesDraggable={false}
          nodesConnectable={false}
          onNodeClick={(_, node) => setSelectedNodeId(Number(node.id))}
          minZoom={0.25}
          maxZoom={1.75}
          colorMode="dark"
        >
          <Background color="#334155" gap={20} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </main>
    </div>
  );
}
