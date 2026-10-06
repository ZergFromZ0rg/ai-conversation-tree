// Posts captured conversations to the configured AI Conversation Tree server.
// Done here, not in the content script: a service worker with host permission
// for the server isn't subject to the chat site's CORS / CSP rules.

const DEFAULTS = { serverUrl: "http://127.0.0.1:8000", token: "", autoSync: false };
const REQUEST_TIMEOUT_MS = 180000; // classifying a long chat on CPU takes a while

async function settings() {
  return { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) };
}

async function ingest(capture) {
  const { serverUrl, token } = await settings();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${serverUrl.replace(/\/+$/, "")}/ingest/conversations`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(capture),
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      const reason =
        response.status === 401
          ? "The server rejected the API token."
          : typeof detail?.detail === "string"
            ? detail.detail
            : `Server returned ${response.status}.`;
      return { ok: false, error: reason };
    }
    return { ok: true, result: await response.json(), serverUrl };
  } catch (error) {
    const reason =
      error.name === "AbortError"
        ? "The server took too long to respond."
        : `Couldn't reach ${serverUrl}. Is the server running?`;
    return { ok: false, error: reason };
  } finally {
    clearTimeout(timeout);
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "ingest") return false;
  ingest(message.capture).then(async (outcome) => {
    if (sender.tab?.id != null) {
      await chrome.action.setBadgeText({ tabId: sender.tab.id, text: outcome.ok ? "✓" : "!" });
      await chrome.action.setBadgeBackgroundColor({
        tabId: sender.tab.id,
        color: outcome.ok ? "#16a34a" : "#dc2626",
      });
    }
    sendResponse(outcome);
  });
  return true; // keep the channel open for the async response
});
