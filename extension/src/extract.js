// Pulls the visible conversation out of a ChatGPT / Claude / Gemini page.
//
// Pure: takes a Document and a location-like object, returns data. content.js
// calls it on the live page; the tests call it on fixture HTML. These sites
// change their markup without notice, so every role has a list of selectors
// tried in order — when a site ships a redesign, this table is the one place
// to update.

(function (root) {
  const SITES = {
    chatgpt: {
      hosts: ["chatgpt.com", "chat.openai.com"],
      idFromPath: /\/c\/([A-Za-z0-9-]+)/,
      titleSuffix: /\s*[|\-–]\s*ChatGPT\s*$/i,
      genericTitle: /^ChatGPT$/i,
      // One element per message, role in an attribute.
      messages(doc) {
        return Array.from(doc.querySelectorAll("[data-message-author-role]")).map((el) => ({
          role: el.getAttribute("data-message-author-role"),
          el,
        }));
      },
      streaming: ['button[data-testid="stop-button"]', 'button[aria-label*="Stop streaming" i]'],
    },
    claude: {
      hosts: ["claude.ai"],
      idFromPath: /\/chat\/([A-Za-z0-9-]+)/,
      titleSuffix: /\s*[|\-–]\s*Claude\s*$/i,
      genericTitle: /^Claude$/i,
      user: ['[data-testid="user-message"]'],
      assistant: [".font-claude-response", ".font-claude-message", '[data-testid="assistant-message"]'],
      streaming: ['[data-is-streaming="true"]', 'button[aria-label*="Stop response" i]'],
    },
    gemini: {
      hosts: ["gemini.google.com"],
      idFromPath: /\/app\/([A-Za-z0-9]+)/,
      titleSuffix: /\s*[|\-–]\s*(Google\s+)?Gemini\s*$/i,
      genericTitle: /^(Google\s+)?Gemini$/i,
      user: ["user-query .query-text", "user-query"],
      assistant: ["model-response message-content", "model-response .markdown", "model-response"],
      streaming: ['button[aria-label*="Stop" i]', "model-response .loading"],
    },
  };

  const textOf = (el) => (el.innerText != null ? el.innerText : el.textContent || "").trim();

  function siteForHost(hostname) {
    return Object.keys(SITES).find((name) => SITES[name].hosts.includes(hostname)) || null;
  }

  // Drop elements nested inside another match, so a wrapper and its inner
  // content node are not both counted as messages.
  function outermost(elements) {
    return elements.filter((el) => !elements.some((other) => other !== el && other.contains(el)));
  }

  function collect(doc, selectors) {
    for (const selector of selectors) {
      const found = outermost(Array.from(doc.querySelectorAll(selector)));
      if (found.length) return found;
    }
    return [];
  }

  function bySelectors(doc, site) {
    const tagged = [];
    collect(doc, site.user).forEach((el) => tagged.push({ role: "user", el }));
    collect(doc, site.assistant).forEach((el) => tagged.push({ role: "assistant", el }));
    // Merge the two lists back into page order.
    tagged.sort((a, b) =>
      a.el.compareDocumentPosition(b.el) & 4 /* b follows a */ ? -1 : 1,
    );
    return tagged;
  }

  function extractConversation(doc, location) {
    const siteName = siteForHost(location.hostname);
    if (!siteName) return { ok: false, reason: "This site isn't supported." };
    const site = SITES[siteName];

    const idMatch = site.idFromPath.exec(location.pathname);
    if (!idMatch) {
      return { ok: false, reason: "Open a saved conversation first (this page has no conversation id yet)." };
    }

    const tagged = site.messages ? site.messages(doc) : bySelectors(doc, site);
    const messages = tagged
      .map(({ role, el }) => ({ role, text: textOf(el) }))
      .filter((message) => message.text);
    if (!messages.length) {
      return { ok: false, reason: "No messages found on the page — its layout may have changed." };
    }

    let title = (doc.title || "").replace(site.titleSuffix, "").trim();
    if (!title || site.genericTitle.test(title)) title = null;

    return {
      ok: true,
      capture: {
        source: siteName,
        externalId: idMatch[1],
        title,
        url: location.href,
        messages,
      },
      isStreaming: site.streaming.some((selector) => doc.querySelector(selector)),
    };
  }

  const api = { extractConversation, siteForHost, SITES };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ActExtract = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
