const TOKEN_STORAGE_KEY = "act.apiToken";

// Hosted deployments gate the API behind a shared token (see README,
// "Hosted deployment"). Local installs leave it unset and nothing changes.
export function getApiToken() {
  try {
    return window.localStorage.getItem(TOKEN_STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export function setApiToken(token) {
  try {
    if (token) window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
    else window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    /* storage unavailable — ignore */
  }
}

function authHeaders() {
  const token = getApiToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// What the server said went wrong. The services answer {"detail": "..."} for a
// missing conversation (404), a rule that was broken (422), or a failed model
// provider (502), and a list of {msg} for a malformed request; fall back to the
// bare status when there is nothing readable.
async function errorMessage(response) {
  try {
    const detail = (await response.clone().json())?.detail;
    if (typeof detail === "string" && detail) return detail;
    if (Array.isArray(detail)) {
      const messages = detail.map((item) => item?.msg).filter(Boolean);
      if (messages.length) return messages.join("; ");
    }
  } catch {
    /* not JSON, or already consumed: use the status */
  }
  return `Request failed: ${response.status}`;
}

async function failIfNotOk(response) {
  if (response.ok) return;
  if (response.status === 401) {
    // Lets the app open Settings on the token field.
    window.dispatchEvent(new Event("act:unauthorized"));
  }
  throw new Error(await errorMessage(response));
}

async function apiRequest(path, init) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...authHeaders(), ...(init?.headers ?? {}) },
    ...init,
  });
  await failIfNotOk(response);
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

export const api = {
  listConversations: () => apiRequest("/conversations"),
  listModels: () => apiRequest("/models"),
  listOllamaLibrary: () => apiRequest("/ollama/library"),
  pullOllamaModel: (model) => streamEvents("/ollama/pull", { model }),
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
  getSummary: (conversationId) => apiRequest(`/conversations/${conversationId}/summary`),
  getLlmSummary: (conversationId, model, apiKey) =>
    apiRequest(`/conversations/${conversationId}/summary/llm`, {
      method: "POST",
      body: JSON.stringify({ model: model ?? null, apiKey: apiKey ?? null }),
    }),
  getContextHealth: (conversationId) =>
    apiRequest(`/conversations/${conversationId}/context-health`),
  searchConversations: (query, limit = 30, conversationId = null) =>
    apiRequest(
      `/search?query=${encodeURIComponent(query)}&limit=${limit}` +
        (conversationId ? `&conversationId=${conversationId}` : ""),
    ),
  downloadReport: async (conversationId, format) => {
    const response = await fetch(`/conversations/${conversationId}/report?format=${format}`, {
      headers: authHeaders(),
    });
    await failIfNotOk(response);
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const filename = disposition.match(/filename="?([^";]+)"?/)?.[1] || `conversation-${conversationId}.${format === "json" ? "json" : "md"}`;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  },
  createConceptLink: (aConceptKey, bConceptKey, kind) =>
    apiRequest("/concept-links", {
      method: "POST",
      body: JSON.stringify({ aConceptKey, bConceptKey, kind }),
    }),
  deleteConceptLink: (linkId) => apiRequest(`/concept-links/${linkId}`, { method: "DELETE" }),
  createTurnEdge: (conversationId, fromTurnId, toTurnId, label, confidence) =>
    apiRequest("/edges", {
      method: "POST",
      body: JSON.stringify({ conversationId, fromTurnId, toTurnId, label, confidence }),
    }),
  updateTurnEdge: (edgeId, label, confidence) =>
    apiRequest(`/edges/${edgeId}`, {
      method: "PATCH",
      body: JSON.stringify({ label, confidence }),
    }),
  deleteTurnEdge: (edgeId) => apiRequest(`/edges/${edgeId}`, { method: "DELETE" }),
  streamTurn: (conversationId, userText, model, apiKey) =>
    streamEvents(`/conversations/${conversationId}/turns/stream`, {
      userText,
      model: model ?? null,
      apiKey: apiKey ?? null,
    }),
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
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  await failIfNotOk(response);
  if (!response.body) {
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
