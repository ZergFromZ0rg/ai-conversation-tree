// Runs on the supported chat sites. Does nothing until asked (popup button) or,
// if the user turned it on, after the page has been quiet for a few seconds.

(function () {
  const AUTO_SYNC_IDLE_MS = 5000;
  let timer = null;
  let lastSent = "";

  function capture() {
    return globalThis.ActExtract.extractConversation(document, location);
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "capture") {
      sendResponse(capture());
    }
  });

  async function autoSync() {
    const { autoSync: enabled } = await chrome.storage.local.get("autoSync");
    if (!enabled) return;
    const result = capture();
    // Wait for streaming to finish: syncing a half-written answer would just
    // be replaced by the finished one a moment later.
    if (!result.ok || result.isStreaming) return;
    const fingerprint = result.capture.externalId + ":" + JSON.stringify(result.capture.messages).length;
    if (fingerprint === lastSent) return;
    lastSent = fingerprint;
    chrome.runtime.sendMessage({ type: "ingest", capture: result.capture });
  }

  new MutationObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(autoSync, AUTO_SYNC_IDLE_MS);
  }).observe(document.body, { childList: true, subtree: true, characterData: true });
})();
