async function apiRequest(path, init) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

export const api = {
  listConversations: () => apiRequest("/conversations"),
  listModels: () => apiRequest("/models"),
  createConversation: (title, model) =>
    apiRequest("/conversations", {
      method: "POST",
      body: JSON.stringify({ title: title ?? null, model: model ?? null }),
    }),
  deleteConversation: (conversationId) =>
    apiRequest(`/conversations/${conversationId}`, { method: "DELETE" }),
  setConversationModel: (conversationId, model) =>
    apiRequest(`/conversations/${conversationId}`, {
      method: "PATCH",
      body: JSON.stringify({ model: model ?? null }),
    }),
  getTurns: (conversationId) => apiRequest(`/conversations/${conversationId}/turns`),
  getGraph: (conversationId) => apiRequest(`/conversations/${conversationId}/graph`),
  getConceptLinks: (conversationId) =>
    apiRequest(`/conversations/${conversationId}/concept-links`),
  getConceptGraph: () => apiRequest("/concepts/graph"),
  createConceptLink: (aConceptKey, bConceptKey, kind) =>
    apiRequest("/concept-links", {
      method: "POST",
      body: JSON.stringify({ aConceptKey, bConceptKey, kind }),
    }),
  deleteConceptLink: (linkId) => apiRequest(`/concept-links/${linkId}`, { method: "DELETE" }),
  streamTurn: (conversationId, userText, model) =>
    streamEvents(`/conversations/${conversationId}/turns/stream`, { userText, model: model ?? null }),
  analyze: (conversationId) =>
    apiRequest(`/conversations/${conversationId}/analyze`, { method: "POST" }),
};

// Reads a POST endpoint's `text/event-stream` body as it arrives, yielding
// each `data: <json>` line's parsed payload. Used for the token-by-token
// turn endpoint; a non-2xx status (validation, missing conversation) throws
// before any events are yielded, same as apiRequest.
async function* streamEvents(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex;
      while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data:"));
        if (dataLine) {
          yield JSON.parse(dataLine.slice("data:".length).trim());
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
