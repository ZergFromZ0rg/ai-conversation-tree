const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");
const { extractConversation, siteForHost } = require("../src/extract.js");

function run(fixture, url) {
  const html = fs.readFileSync(path.join(__dirname, "fixtures", fixture), "utf8");
  const dom = new JSDOM(html, { url });
  // jsdom doesn't implement innerText; textContent is close enough for fixtures.
  Object.defineProperty(dom.window.HTMLElement.prototype, "innerText", {
    get() {
      return this.textContent;
    },
  });
  return extractConversation(dom.window.document, dom.window.location);
}

test("ChatGPT: roles come from attributes, title loses the suffix", () => {
  const result = run("chatgpt.html", "https://chatgpt.com/c/abc-123");
  assert.equal(result.ok, true);
  assert.equal(result.capture.source, "chatgpt");
  assert.equal(result.capture.externalId, "abc-123");
  assert.equal(result.capture.title, "Cat diet");
  assert.deepEqual(
    result.capture.messages.map((m) => m.role),
    ["user", "assistant", "user", "assistant"],
  );
  assert.match(result.capture.messages[1].text, /obligate carnivores/);
  assert.equal(result.isStreaming, false);
});

test("ChatGPT: project-scoped URLs still yield the conversation id", () => {
  const result = run("chatgpt.html", "https://chatgpt.com/g/g-xyz/c/def-456");
  assert.equal(result.capture.externalId, "def-456");
});

test("Claude: user/assistant interleaved in page order, streaming detected", () => {
  const result = run("claude.html", "https://claude.ai/chat/1b2c-3d4e");
  assert.equal(result.ok, true);
  assert.equal(result.capture.source, "claude");
  assert.equal(result.capture.title, "Python decorators");
  assert.deepEqual(
    result.capture.messages.map((m) => m.role),
    ["user", "assistant", "user", "assistant"],
  );
  assert.equal(result.isStreaming, true);
});

test("Gemini: custom elements, generic title becomes null", () => {
  const result = run("gemini.html", "https://gemini.google.com/app/9f8e7d");
  assert.equal(result.ok, true);
  assert.equal(result.capture.source, "gemini");
  assert.equal(result.capture.externalId, "9f8e7d");
  assert.equal(result.capture.title, null);
  assert.equal(result.capture.messages.length, 4);
  assert.equal(result.capture.messages[0].text, "Explain HTTP caching");
});

test("a new chat with no id yet is rejected with guidance", () => {
  const result = run("chatgpt.html", "https://chatgpt.com/");
  assert.equal(result.ok, false);
  assert.match(result.reason, /saved conversation/);
});

test("a page whose layout the selectors no longer match says so", () => {
  const dom = new JSDOM("<html><body><p>nothing</p></body></html>", { url: "https://claude.ai/chat/abc" });
  const result = extractConversation(dom.window.document, dom.window.location);
  assert.equal(result.ok, false);
  assert.match(result.reason, /layout may have changed/);
});

test("unsupported hosts are refused", () => {
  assert.equal(siteForHost("example.com"), null);
  const dom = new JSDOM("<html></html>", { url: "https://example.com/c/1" });
  assert.equal(extractConversation(dom.window.document, dom.window.location).ok, false);
});
