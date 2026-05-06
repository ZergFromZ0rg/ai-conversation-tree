import type { ConversationPayload, ConversationSummary } from "../types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function createConversation(title?: string): Promise<number> {
  const payload = await request<{ conversationId: number }>("/conversations", {
    method: "POST",
    body: JSON.stringify({ title: title ?? "New Conversation" })
  });
  return payload.conversationId;
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const payload = await request<{ conversations: ConversationSummary[] }>("/conversations");
  return payload.conversations;
}

export async function getConversation(conversationId: number): Promise<ConversationPayload> {
  return request<ConversationPayload>(`/conversations/${conversationId}`);
}

export async function sendChatMessage(conversationId: number, userText: string): Promise<ConversationPayload> {
  return request<ConversationPayload>("/chat", {
    method: "POST",
    body: JSON.stringify({ conversationId, userText })
  });
}

export async function reclassifyGraph(): Promise<{ nodes: number; edges: number }> {
  return request<{ nodes: number; edges: number }>("/reclassify", {
    method: "POST"
  });
}
