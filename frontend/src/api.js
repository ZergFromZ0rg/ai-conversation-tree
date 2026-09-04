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
  sendTurn: (conversationId, userText, model) =>
    apiRequest(`/conversations/${conversationId}/turns`, {
      method: "POST",
      body: JSON.stringify({ userText, model: model ?? null }),
    }),
  analyze: (conversationId) =>
    apiRequest(`/conversations/${conversationId}/analyze`, { method: "POST" }),
};
