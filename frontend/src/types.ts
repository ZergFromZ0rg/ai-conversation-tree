export type SemanticParent = [number, string, number];

export type Turn = {
  id: number;
  conversationId: number;
  userText: string;
  aiText: string;
  timestamp: string;
  root: boolean;
  timelineParent: number | null;
  conceptIds: number[];
  semanticParents: SemanticParent[];
};

export type ConversationSummary = {
  id: number;
  title: string | null;
  createdAt: string;
  updatedAt: string;
};

export type GraphNode = {
  id: number;
  userText: string;
  aiText: string;
  conceptIds: number[];
  root: boolean;
  timelineParent: number | null;
};

export type GraphEdge = {
  from: number;
  to: number;
  type: string;
  confidence: number;
};

export type ConversationPayload = {
  conversationId: number;
  turns: Turn[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  turnId?: number;
  aiText?: string;
};
