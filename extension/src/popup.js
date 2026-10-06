const $ = (id) => document.getElementById(id);
const DEFAULTS = { serverUrl: "http://127.0.0.1:8000", token: "", autoSync: false };
const LOCAL_ORIGINS = ["http://127.0.0.1", "http://localhost"];

let capture = null;

function setStatus(text, kind = "") {
  $("status").textContent = text;
  $("status").className = kind;
}

async function loadSettings() {
  const saved = { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) };
  $("serverUrl").value = saved.serverUrl;
  $("token").value = saved.token;
  $("autoSync").checked = saved.autoSync;
  return saved;
}

async function saveSettings() {
  const serverUrl = $("serverUrl").value.trim().replace(/\/+$/, "") || DEFAULTS.serverUrl;
  let origin;
  try {
    origin = new URL(serverUrl).origin;
  } catch {
    setStatus("That server URL isn't valid.", "error");
    return;
  }
  // Localhost is pre-approved in the manifest; any other server (a hosted
  // deployment) is requested here, from the click, as Chrome requires.
  const isLocal = LOCAL_ORIGINS.some((local) => origin.startsWith(local));
  if (!isLocal && !(await chrome.permissions.request({ origins: [`${origin}/*`] }))) {
    setStatus("Permission to reach that server was declined.", "error");
    return;
  }
  await chrome.storage.local.set({
    serverUrl,
    token: $("token").value.trim(),
    autoSync: $("autoSync").checked,
  });
  setStatus("Settings saved.", "ok");
}

async function inspectTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  let reply = null;
  try {
    reply = tab?.id != null ? await chrome.tabs.sendMessage(tab.id, { type: "capture" }) : null;
  } catch {
    // No content script here: not a supported site (or the tab predates install).
  }
  if (!reply) {
    $("site").textContent = "Open a ChatGPT, Claude, or Gemini conversation to use this.";
    return;
  }
  if (!reply.ok) {
    $("site").textContent = reply.reason;
    return;
  }
  capture = reply.capture;
  const exchanges = Math.floor(reply.capture.messages.length / 2);
  $("site").textContent = `${reply.capture.source} · ${reply.capture.title || "untitled"} · ~${exchanges} exchange${exchanges === 1 ? "" : "s"}`;
  $("sync").disabled = false;
  if (reply.isStreaming) setStatus("The reply is still streaming — wait for it to finish.");
}

async function sync() {
  $("sync").disabled = true;
  setStatus("Sending…");
  const outcome = await chrome.runtime.sendMessage({ type: "ingest", capture });
  $("sync").disabled = false;
  if (!outcome?.ok) {
    setStatus(outcome?.error || "Something went wrong.", "error");
    return;
  }
  const { addedTurns, turnCount, rebuilt, conversationId } = outcome.result;
  setStatus(
    rebuilt
      ? `Chat changed — rebuilt with ${turnCount} turns.`
      : addedTurns
        ? `Added ${addedTurns} new turn${addedTurns === 1 ? "" : "s"} (${turnCount} total).`
        : `Already up to date (${turnCount} turns).`,
    "ok",
  );
  const link = $("open");
  link.href = `${outcome.serverUrl.replace(/\/+$/, "")}/ui?c=${conversationId}`;
  link.hidden = false;
}

$("save").addEventListener("click", saveSettings);
$("sync").addEventListener("click", sync);
void loadSettings().then(inspectTab);
