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
  createConversation: (title) =>
    apiRequest("/conversations", {
      method: "POST",
      body: JSON.stringify({ title: title ?? null }),
    }),
  deleteConversation: (conversationId) =>
    apiRequest(`/conversations/${conversationId}`, { method: "DELETE" }),
  getTurns: (conversationId) => apiRequest(`/conversations/${conversationId}/turns`),
  getGraph: (conversationId) => apiRequest(`/conversations/${conversationId}/graph`),
  sendTurn: (conversationId, userText) =>
    apiRequest(`/conversations/${conversationId}/turns`, {
      method: "POST",
      body: JSON.stringify({ userText }),
    }),
  analyze: (conversationId) =>
    apiRequest(`/conversations/${conversationId}/analyze`, { method: "POST" }),
};
