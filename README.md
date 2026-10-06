# AI Conversation Tree

`AI Conversation Tree` turns chat transcripts into a map you can search and
audit. It decides how each turn relates to the ones before it (continuation,
branch, related, or a fresh topic), links concepts that recur across your
conversations, and warns when a long chat has probably lost its own context.

Chats get in two ways: type into the built-in viewer (local models, OpenAI,
Anthropic, Gemini, or a stub), or **capture the conversations you already have
in ChatGPT, Claude, and Gemini with the browser extension**.

It runs on your machine by default (SQLite, local embedding models, no
accounts). It is a technically solid, explainable portfolio project, not a
hosted product — though a Docker setup is included if you want one.

![The conversation graph drawer: a seven-turn imported ChatGPT chat with its turns linked by continuation and related edges, a context-loss warning in the transcript, and an "also discussed elsewhere" panel pointing at a Gemini chat.](docs/images/conversation-graph.png)

## Five-Minute Demo

This walks through everything with no API key and no browser extension. It
loads four chats from three sources the same way the extension would.

```bash
# 1. backend + viewer (see Local Setup for the one-time install)
AI_CONVERSATION_TREE_STUB_LLM=1 AI_CONVERSATION_TREE_CONTEXT_TOKENS=400 \
  venv/bin/python -m uvicorn api:app

# 2. in another terminal: load the demo workspace through the ingest API
venv/bin/python scripts/seed_demo.py
```

Open <http://127.0.0.1:8000/ui>, then:

1. **Read a chat with a graph.** Pick *Planning a Postgres migration* (ChatGPT
   badge). Press `Cmd/Ctrl+G` to open the graph: a seven-turn chat that
   splits into threads, each edge labelled and scored.
2. **See context-loss detection.** In the transcript, turn 5 ("*As I said*, we
   cannot have any downtime…") and turn 6 (a question asked verbatim in turn 3)
   carry an amber warning, and a dashed line marks where earlier messages fall
   outside the model's context window. (The demo shrinks the window to 400
   tokens so a short chat crosses it; the default is 8,000.)
3. **Search across every chat.** Press `Cmd/Ctrl+K`, type `index concurrently`.
   Hits from a Gemini chat and the ChatGPT chat appear together, grouped by
   conversation, with matched words highlighted. `Enter` jumps straight to that
   turn.
4. **Follow a cross-chat link.** The graph drawer's *Also discussed elsewhere*
   panel shows the Gemini indexing chat. Open `Map` for the whole workspace:
   the two Postgres chats are linked, and so are the two decorator chats.
5. **Summarise.** `Summary` shows the local summary and the context-health
   report; with a real model selected, *Write with the selected model*
   replaces the summary with a model-written one.

![Search across chats: results grouped by conversation with source badges, question/answer tags, turn positions, and highlighted matches.](docs/images/search.png)

![Summary and context health: the structured summary, an "At risk" verdict, token pressure against the context window, and the flagged turns, each one click from the transcript.](docs/images/context-health.png)

## Architecture

```mermaid
flowchart LR
    subgraph sources[Where chats come from]
        EXT["Browser extension<br/>ChatGPT · Claude · Gemini"]
        UI["Built-in chat viewer"]
    end
    EXT -- "POST /ingest/conversations" --> API
    UI -- "turns / stream" --> API
    API["FastAPI"] --> CLS["Classifier<br/>embeddings + discourse + cross-encoder"]
    CLS --> DB[("SQLite<br/>turns · edges · concepts · embeddings")]
    DB --> CON["Concept linker<br/>cross-conversation"]
    DB --> SRCH["Hybrid search"]
    DB --> CTX["Context-health analysis"]
    API --> VIEW["Viewer: chat · graph · map · search"]
    CON --> VIEW
    SRCH --> VIEW
    CTX --> VIEW
```

## What It Does

For each new turn, the system decides whether it is:

- a `continuation` of an existing topic
- a `branch` off a previous answer or subthread
- `related` to one or more prior topics
- effectively unrelated, in which case it starts a new root thread

The result is stored as:

- conversation records
- turns
- semantic edges
- concept assignments
- concept links across conversations
- embeddings

and rendered as a chat with a graph drawer.

Separately, once a turn lands or a conversation is re-analyzed, every concept in
that conversation is scored against the concepts of every *other* conversation.
Close pairs become `conceptLinks`, surfaced in the drawer as "also discussed
elsewhere" and over `GET /concepts/graph`.

Response generation runs through a stub, a local `Ollama` model, `OpenAI`,
`Anthropic`, or `Gemini`. The environment sets the default; the UI and API can
override it per conversation. Ollama, OpenAI, and Anthropic support native
incremental streaming; Gemini currently falls back to one response chunk.

## Core Approach

### Immediate Previous Turn

The immediate previous turn gets the strongest classifier. It combines three
signal sources:

- **embedding similarity** — `all-MiniLM-L6-v2` cosine between the new user
  message and the previous turn (`user + " " + ai`)
- **discourse features** — regex pattern families (clarification, reference,
  bare pronoun, forward / "how do I", comparison, lateral shift) plus lexical
  and content-term overlap
- **cross-encoder scores** — `cross-encoder/ms-marco-MiniLM-L-6-v2` run once
  per candidate label with a label-specific prompt

Discourse features decide which labels are *eligible*; the embedding and
cross-encoder signals set the confidence. A label is only scored when
something anchors it to the previous turn:

- `branch` — a clarification / reference / pronoun cue **with** a trace of
  shared topic under it (so a bare "why …?" after a hard topic switch is not a
  branch off nothing), or a confident cross-encoder `branch` score (the new
  question is answerable straight from the last answer)
- `continuation` — a "how do I … next" marker on an established topic, a
  comparison follow-up, a bare pronoun pointing back, or the message just
  staying on topic (high raw similarity to the turn or its answer) with no
  lateral-shift marker
- `related` — a "what about X instead" lateral shift, two parallel `what is X`
  questions about the same area, or real surface anchoring (shared content
  terms plus topic overlap) without a shift marker

Each eligible label's confidence blends its heuristic score, its cross-encoder
score, and the embedding similarities. The highest wins if it clears
`edgeConfidenceThreshold` (0.45); otherwise the turn gets **no edge** and
starts a new root.

`eval_immediate_previous.py` (driven by `eval_cases.json`) grades this on 27
cases across the four outcomes, plus 3 `knownGap` cases the bi-encoder can't
resolve — anaphora with no shared terms, acronyms it doesn't know — which are
reported but not graded.

### Older Prior Turns

Older-turn linking uses batched vector retrieval first, then classification.

Current flow:

1. score earlier turn embeddings in one NumPy matrix operation, minus a small
   decay for older turns
2. keep only the top few candidates
3. classify each selected older link as `continuation` or `related`

At this scale a direct scan is cheap and avoids the concept-centroid layer,
whose average embedding is a poor representative once a concept has drifted.

### Cross-Conversation Concept Links

A concept is the set of turns that share a `conceptId` inside one conversation.
`conceptIndex.py` compares concepts *across* conversations:

1. group every turn embedding by `(conversationId, conceptId)` and L2-normalise
2. score two concepts as the mean of the top-3 pairwise cosine similarities
   between their member turns — top-k, not max (one stray turn pair should not
   forge a link) and not the full mean (which dilutes a broad concept)
3. use normalized concept centroids to shortlist candidate pairs, then apply
   exact top-3 member scoring; `>= 0.66` is a `same` link, `>= 0.52` is
   `related`; cap 3 links per concept
4. concepts with no turn of at least four distinct word tokens are skipped, so
   greetings and acknowledgements do not link

`conceptId`s are reassigned from zero on every re-analysis, so each concept
also carries a stable `conceptKey` (`turnConcepts.conceptKey`): a re-analysed
concept inherits the key of the old concept it overlaps most (Jaccard >= 0.5,
matched greedily), and anything new gets a fresh key. `conceptLinks` is keyed
by this pair of `conceptKey`s, not by `(conversationId, conceptId)` — so a link
keeps pointing at the same concept across re-analysis. `auto` links are still
deleted and rebuilt on every change — debounced on the send path, forced on
re-analysis. A `manual` link (`POST /concept-links`) is left alone by that
rebuild and only disappears if one of its two concepts stops existing (a
concept that did not survive re-analysis, or its conversation was deleted) —
`pruneOrphanConceptLinks` runs in both of those transactions.
Thresholds were tuned against `eval_concept_links.py`: with
`all-MiniLM-L6-v2` on short questions, "same topic, different wording" pairs
land around 0.55-0.65 while the nearest false positives sit below 0.48.
Genuinely adjacent topics the model cannot lift out of the noise floor stay
unlinked by design.

**Tried and rejected: blending in the cross-encoder.** The immediate-turn
classifier gets real value from `cross-encoder/ms-marco-MiniLM-L-6-v2` (see
"Immediate Previous Turn" above), so it seemed worth trying for concept
scoring too. Measured against the same pairs `eval_concept_links.py`'s
thresholds were tuned on (full questions and short label-style text alike): it
reproduces cosine's
top-3 ranking almost exactly (0.65-0.96) and returns ~0.000 for every other
pair — including the ones cosine alone can't separate from noise, like
"train a neural network" vs. "overfitting in machine learning" (0.239
cosine, 0.00003 cross-encoder) or "REST API design" vs. "GraphQL API design"
(0.405 cosine, 0.012 cross-encoder). It's a query-passage reranker; two full
questions from unrelated conversations aren't the (short query, passage)
shape it was trained on, so it just strongly confirms whatever cosine was
already confident about and adds no signal in the gray zone — not worth the
extra model call per candidate pair.

Each concept is labelled by its most distinctive terms — term frequency
(across the concept's own turns) times inverse concept frequency (rarer
across the workspace scores higher), using a label-specific stopword list
(`conceptIndex._labelStopwords`, broader than the classifier's own) — rather
than by its first question verbatim, so "python, decorator" instead of "how
do decorators work in python". A concept with no content terms at all (a
greeting) falls back to that trimmed first question.

### Important Design Principle

Topic identity should dominate continuation decisions.

Discourse features are used as supporting evidence, not as ground truth. This reduces false positives where generic follow-up phrasing looks like continuation even when the topic changed.

## Tech Stack

Backend:

- `Python`
- `FastAPI`
- `SQLite`
- `Pydantic`
- `NumPy`
- `sentence-transformers`
- `CrossEncoder`
- `PyTorch`
- `Ollama` (local), `OpenAI`, `Anthropic`, or `Gemini` for response generation;
  a stub mode needs neither

Frontend:

- `React` + `Vite`
- `React Flow` for the graph canvas
- `Dagre` for hierarchical graph layout

## Project Structure

Backend:

- `api.py`
  - HTTP routes
- `chatService.py`
  - orchestration, provider calls, graph payload serialization
- `graphService.py`
  - `ConversationGraph` model, classification logic, concept retrieval, reanalysis
- `conceptIndex.py`
  - cross-conversation concept scoring, link rebuild, concept labels
- `graphStore.py`
  - per-conversation graph cache (LRU) and per-conversation write lock
- `vectorStore.py`
  - optional PostgreSQL/pgvector HNSW index for scalable turn retrieval, with
    automatic fallback to local NumPy search
- `analysisService.py`
  - structured and LLM-written summaries, hybrid cross-chat search, and
    Markdown/JSON report generation
- `ingestService.py`
  - turns captured role-tagged messages into turns, matches a chat to an
    earlier import, and classifies only what is new
- `contextService.py`
  - context-loss detection (pure scoring function plus a DB wrapper)
- `db.py`
  - `SQLite` schema and helpers
- `models.py`
  - the `TurnModel` dataclass

Elsewhere:

- `extension/` — the Manifest V3 browser extension and its tests
- `tests/` — the pytest suite
- `scripts/seed_demo.py` — loads the demo workspace through the ingest API
- `Dockerfile`, `docker-compose.yml` — hosted deployment

Frontend (`frontend/src/`):

- `App.jsx`
  - shell: conversation state, layout
- `api.js`
  - fetch wrappers
- `useConversation.js`
  - hook owning one conversation's turns, graph, and concept links, plus the send / analyze / refresh actions
- `components/`
  - `ConversationSidebar`, `ChatTranscript`, `Composer` (with the model picker),
    `GraphRail`, `GraphDrawer` (with the "also discussed elsewhere" panel),
    `TurnGraph`, `WorkspaceMap`

## Persistence Model

Everything is saved locally in `.data/conversationTree.db` (`SQLite`, WAL mode)
by default. Set `AI_CONVERSATION_TREE_DB` to choose another path. Runtime data
is ignored by Git; the old root-level `conversationTree.db` is retained only
as a legacy sample database.

- `conversations` — `model` column holds the conversation's default response
  model (`stub`, `ollama:<name>`, or `openai:<name>`); null means use the
  environment default. Imported chats also carry `source` (`chatgpt` /
  `claude` / `gemini`), `externalId` (the site's own conversation id), and
  `sourceUrl`; `(source, externalId)` is unique, which is how a re-sync finds
  its conversation
- `turns`
- `semanticEdges` — each edge carries an `origin` of `auto` (classifier) or `manual` (hand-created via `POST /edges`)
- `turnConcepts` — `conceptKey` is a stable per-concept identity carried across
  re-analysis by turn overlap; the integer `conceptId` is only a within-
  conversation label
- `conceptLinks` — similarity links between two `conceptKey`s in *different*
  conversations; the pair is stored once in a canonical order (no FK — a
  key's rows live in `turnConcepts`; `pruneOrphanConceptLinks` deletes a link
  once either key no longer exists), with an `origin` of `auto` (rebuilt on
  every change) or `manual` (hand-created via `POST /concept-links`)
- `turnEmbeddings` — one `float32` vector per turn, stored as a `BLOB`
  (`np.frombuffer` / `.tobytes()`), not JSON text

The first request for a conversation rebuilds its in-memory `ConversationGraph`
from persisted rows; `graphStore` then keeps it cached. Each cache entry carries
the conversation's SQLite `updatedAt` version and is reloaded when another
backend process changes that conversation, so multiple workers do not retain
stale graph state. A turn is written in one SQLite transaction (turn,
embedding, classifier edges, concept memberships — all or nothing), and if a
mutation fails part-way the cached graph is discarded and reloaded from SQLite,
so memory never holds a turn the database doesn't. Because every write bumps
`updatedAt`, the next request after a write reloads the graph; that is
deliberate, as it is what keeps other workers from serving stale state.

## API

### Model Endpoints

- `GET /models` — installed `Ollama` tags (queried live from `/api/tags`), the
  configured `OpenAI` model, and `stub`, plus the current default and whether
  `Ollama` is reachable

### Conversation Endpoints

- `POST /conversations` — optional `model` sets the conversation default
- `GET /conversations` — each entry carries `source` / `sourceUrl` when the
  chat was imported by the extension (null otherwise)
- `GET /conversations/{conversationId}`
- `PATCH /conversations/{conversationId}` — set the conversation's default
  response model (`{"model": "ollama:<name>"}`); validated against the same
  `stub` / `ollama:<name>` / `openai:<name>` shape
- `DELETE /conversations/{conversationId}`

### Turn Endpoints

- `POST /conversations/{conversationId}/turns` — optional `model` overrides the
  provider for this turn and becomes the conversation's new default; a provider
  failure returns `502`. If this is the conversation's first turn and it has no
  title yet, the title becomes the user message (collapsed, trimmed to 60 chars
  on a word boundary) — a title given at creation (`POST /conversations`) is
  never overwritten
- `POST /conversations/{conversationId}/turns/stream` — same request body and
  side effects (model override, auto-title, concept relink), but the response
  is `text/event-stream`: any number of `{"type": "delta", "text": "..."}`
  events as the reply is generated, then one
  `{"type": "done", "turnId", "aiText", "turns", "nodes", "edges"}` carrying
  the same payload the blocking endpoint returns. A provider or persistence
  failure arrives as `{"type": "error", "message": "..."}` instead of an HTTP
  error status — by the time that can happen the response has already started
  with `200`, so there's no status left to change. `Ollama` and the stub
  stream incremental chunks; Gemini currently sends its whole reply as one
  `delta`
- `GET /conversations/{conversationId}/turns`

### Ingest Endpoint

- `POST /ingest/conversations` — create or update a conversation captured
  elsewhere. Body: `{"source": "chatgpt"|"claude"|"gemini"|"other",
  "externalId": "...", "title"?, "url"?, "messages": [{"role": "user"|"assistant",
  "text": "..."}]}`. See "Browser Extension" for the exact semantics. Returns
  `{conversationId, created, addedTurns, rebuilt, turnCount}`; `422` if no
  complete user/assistant exchange is present

### Analysis Endpoints

- `GET /conversations/{conversationId}/summary` — deterministic local summary
  with turn count, thread count, relationship count, and topic terms
- `POST /conversations/{conversationId}/summary/llm` — the same summary with
  its prose written by a model (`{"model"?, "apiKey"?}`; defaults to the
  conversation's model). Falls back to the local summary, with a
  `fallbackReason`, on the stub model or a provider error. Cached per
  conversation state and model
- `GET /conversations/{conversationId}/context-health` — context-loss report
  (see "Context-Loss Detection")
- `GET /search?query=...&limit=20&conversationId=` — hybrid search across all
  conversations (or one). Each hit carries `snippet`, `matchedTerms`,
  `matchIn` (`question` | `answer`), `turnId`, `turnCount`, `source`, and
  `score`
- `GET /conversations/{conversationId}/report?format=markdown|json` — download
  an analysis report containing the summary, topics, turns, and semantic edges

### Graph Endpoints

- `POST /conversations/{conversationId}/analyze` — recomputes the classifier
  (`auto`) edges and every turn's root / concept assignments in a single
  transaction; `manual` edges are left in place, then concept links are rebuilt
- `GET /conversations/{conversationId}/graph`
- `GET /conversations/{conversationId}/concept-links` — this conversation's
  concepts, each grouped with the other conversations/concepts it links to
  (label, title, score, `same` / `related`); shaped for the drawer

### Concept Link Endpoints

- `GET /concepts/graph` — the whole workspace: every concept a node
  (`conversationId`, `conceptId`, `conceptKey`, `label`, `turnCount`,
  `conversationTitle`), every link an edge (`id`, `a`, `b`, `score`, `kind`,
  `origin`); each endpoint of an edge also carries its `conceptKey`
- `POST /concepts/relink` — rebuild every conversation's `auto` concept links
  (for a threshold change or to repair drift); returns `{"linkCount": n}`
- `POST /concept-links` — hand-link two concepts by `conceptKey`:
  `{"aConceptKey", "bConceptKey", "kind": "same" | "related"}`; `origin` is
  `manual`, score fixed at `1.0`; `404` if either key is unknown
- `DELETE /concept-links/{linkId}`

### Edge Correction Endpoints

- `POST /edges`
- `PATCH /edges/{edgeId}`
- `DELETE /edges/{edgeId}`

`POST /edges` validates that both turns exist in the conversation, that
`fromTurnId` is earlier than `toTurnId`, that `label` is one of
`continuation` / `branch` / `related`, and that `confidence` is in `[0, 1]`.
Hand-created edges are marked `origin: "manual"` and are preserved when a
conversation is re-analyzed; classifier edges (`origin: "auto"`) are rebuilt.

### Debug Endpoint

- `POST /debug/immediate`

## Stable Graph Output

Graph endpoints return stable JSON:

```json
{
  "nodes": [
    {
      "id": 0,
      "userText": "what are cats?",
      "aiText": "...",
      "conceptIds": [0],
      "root": true,
      "timelineParent": null
    }
  ],
  "edges": [
    {
      "id": 1,
      "fromTurnId": 0,
      "toTurnId": 1,
      "label": "related",
      "confidence": 0.696,
      "origin": "auto"
    }
  ]
}
```

Semantics:

- green edge = `continuation`
- orange edge = `branch`
- blue bidirectional edge = `related`
- orange node border = `root`

`GET /concepts/graph` returns the cross-conversation view:

```json
{
  "nodes": [
    {
      "conversationId": 1,
      "conceptId": 0,
      "conceptKey": "9f2c…",
      "label": "characteristic, key, cat",
      "turnCount": 2,
      "conversationTitle": "Cats"
    }
  ],
  "edges": [
    {
      "id": 7,
      "a": { "conversationId": 1, "conceptId": 0, "conceptKey": "9f2c…" },
      "b": { "conversationId": 4, "conceptId": 2, "conceptKey": "7ab1…" },
      "score": 0.71,
      "kind": "same",
      "origin": "auto"
    }
  ]
}
```

## Interface

The viewer at `/ui` is a normal chat: a conversation list on the left, the
transcript in the middle, a composer at the bottom.

![The chat viewer: conversation list with source badges on the left, an imported chat in the middle with its context-window horizon and a context-loss warning, model picker and composer at the bottom.](docs/images/chat-view.png)

The composer has a model picker populated from `GET /models`. Choosing a model
sends it with each turn and stores it as the conversation's default; a new chat
inherits the current selection, and switching conversations adopts that
conversation's stored model. The last choice is kept in `localStorage`.

The graph lives in a drawer on the right:

- a pill on the right edge shows the turn count; click it (or press
  `Cmd/Ctrl+G`) to slide the graph drawer open
- the graph is laid out top-to-bottom by `Dagre` along the primary-parent spine;
  branch and related links are drawn as secondary curves
- clicking a graph node scrolls the transcript to that turn and highlights it
- the drawer header has `Refresh` and `Reanalyze`
- `Edit edges`, above the graph, switches to hand-editing mode for this
  conversation's own edges (`POST`/`PATCH`/`DELETE /edges`) — a label picker
  (`continuation`/`branch`/`related`) next to it sets what the next pair
  creates. Click a turn, then another: if no manual edge exists between them
  yet, this creates one with the current label; if one already does, this
  relabels it instead of creating a duplicate — the same two-click gesture
  covers both. Click a manually-created edge to remove it; clicking a
  classifier (`auto`) edge in this mode does nothing but say so. Outside edit
  mode, clicking a node behaves as above (select and scroll)
- below the graph, an "also discussed elsewhere" panel lists the other
  conversations that share the selected turn's concept(s); each is a button that
  switches to that conversation

The `Map` button in the chat header opens a full-screen workspace map: every
concept across every conversation as a node, laid out in a grid clustered and
coloured by conversation, with the cross-conversation concept links as edges.
Clicking a concept switches to that conversation.

The chat header also provides:

- `Search` (`Cmd/Ctrl+K`) — hybrid search across all stored conversations;
  `Enter` jumps to the matching turn
- `Summary` — the active conversation's summary (local, or written by the
  selected model) plus its context-health report
- `Export` — downloads a Markdown analysis report; JSON reports are available
  through the report API

![The workspace map: concepts from four conversations clustered by colour, joined by cross-conversation concept links.](docs/images/workspace-map.png)

Every row of clusters has a genuinely empty gap above and below it (nothing is
ever drawn there); an edge between two nodes that a straight line would cut
through other clusters to reach is routed through that gap instead — down or
up out of its own row, across, and into the target's row from its own gap. Two
concepts more than one row apart share no single gap, so that case adds one
more hop: out to a fixed vertical lane past the last column, then down to the
target's row. Same-conversation or same-row-adjacent-column edges skip all of
this — a straight line already doesn't cross anything.

`Link concepts` in the map toolbar switches to hand-linking mode: click one
concept, then another, to `POST /concept-links` a `manual` link between them
(the toggle next to it picks `related` or `same`); a pinned link is drawn in a
third colour and labelled "pinned". Clicking a pinned link removes it; clicking
an `auto` link in this mode does nothing — those come from the classifier, not
from a click.

Assistant replies stream in token by token via
`POST /conversations/{conversationId}/turns/stream` (`text/event-stream`); the
composer shows a typing indicator until the first chunk arrives, then grows
the reply in place. Classification and persistence still happen only once the
full reply is in — the graph, concept links, and title update right after the
stream ends, same as the blocking path.

The UI follows the browser's light/dark preference.

## Browser Extension

`extension/` is a Manifest V3 extension (Chrome and other Chromium browsers)
that captures the conversation open in a ChatGPT, Claude, or Gemini tab and
sends it to your server.

**Install**

1. Open `chrome://extensions`, enable *Developer mode*, choose *Load unpacked*,
   and select the `extension/` folder.
2. Open a saved conversation on `chatgpt.com`, `claude.ai`, or
   `gemini.google.com`, click the extension icon, and press *Send this chat to
   the tree*. The popup reports how many turns were added and links to the
   conversation in the viewer.
3. The popup's *Settings* hold the server URL (default
   `http://127.0.0.1:8000`) and an API token if the server requires one.
   Pointing it at a non-local server asks Chrome for permission to reach that
   origin. *Sync automatically after each reply* is off by default; when on,
   the extension syncs once the page has been quiet for five seconds and no
   reply is streaming.

**What gets sent.** Only the messages of the tab you act on (or, with auto-sync
on, of supported chat pages), only to the server URL you configured. Nothing is
sent until you click the button or enable auto-sync.

**Ingestion semantics** (`ingestService.py`)

- Messages are paired into turns: consecutive user messages merge into one
  prompt, consecutive assistant messages into one reply. A leading assistant
  greeting and a trailing prompt with no reply yet are dropped — the next sync
  picks up the finished turn.
- A chat is identified by `(source, externalId)`. Re-syncing a chat that grew
  appends only the new turns, so the cost is proportional to what changed. A
  chat that is unchanged is a no-op.
- If the stored history *disagrees* with what arrives (an edited prompt, a
  regenerated answer), the conversation is rebuilt from the incoming history
  rather than guessing which turns survive; the response says `rebuilt: true`.
- Imported turns go through the same classifier, concept assignment, and
  cross-chat linking as typed ones, so they show up in the graph, the map,
  search, and context-health immediately. Classifying on CPU takes roughly a
  second or two per turn.

**Fragile by nature.** The extractors read each site's DOM, and those sites
change their markup without notice. The selectors live in one table at the top
of `extension/src/extract.js` (several fallbacks per role). The extractor is
unit-tested against fixture HTML for all three sites, but **the fixtures are
hand-written approximations of each site's structure, not captures of the live
pages** — if a site ships a redesign the popup says "layout may have changed"
and that table is the one place to update.

```bash
cd extension && npm install && npm test   # extractor tests (jsdom)
```

## Search

`GET /search` (`Cmd/Ctrl+K` in the viewer) is hybrid: embedding cosine finds
turns that mean the same thing in different words; a lexical pass rewards turns
containing the query's own words, so exact phrases and identifiers are found
too. Question hits get a small boost over answer hits.

The viewer groups hits by conversation (with the source badge), labels each as
a question or answer match with its position ("turn 3 of 7"), shows a snippet
around the match with the words highlighted, and can scope to the current chat.
`↑`/`↓` move through results, `Enter` opens the conversation and scrolls to the
exact turn.

## Context-Loss Detection

Long chats fail quietly: the model stops seeing something said fifty turns ago.
`contextService.py` looks for the tells in the conversation's own structure —
deterministic, no model call:

| Signal | Fires when | Weight |
| --- | --- | --- |
| `reask` | the user repeats an earlier question (embedding cosine ≥ 0.80, ≥ 0.90 for the immediately preceding turn) | 0.50 |
| `memoryCue` | the user points back: "as I said", "you forgot", "remember that…", "go back to…" | 0.45 |
| `farReach` | a turn is semantically tied to one ≥ 4 turns back **and** the text between them exceeds the context window | 0.60 |
| `longRange` | same, but still inside the window | 0.25 |

A turn's score combines its signals as `1 − Π(1 − w)`; ≥ 0.60 is `high`, ≥ 0.35
`medium`. The conversation gets an overall verdict, a token estimate
(characters ÷ 4) against the assumed window, and a *horizon*: the earliest turn
that still fits in the window counting back from the newest one. The transcript
draws a dashed line there and an amber note under each flagged turn; the
Summary panel lists them, each one click from the turn.

The window defaults to 8,000 tokens and is set with
`AI_CONVERSATION_TREE_CONTEXT_TOKENS`. It is an assumption about how much a
model attends to well, not a measurement of any particular model, and these
signals indicate *risk*, not proof — a repeated question can be deliberate.

## Hosted Deployment

For an always-on server (so the extension can sync from any machine):

```bash
export AI_CONVERSATION_TREE_API_TOKEN=$(openssl rand -hex 24)
docker compose up -d --build
```

- The image bakes in both models, so the container starts offline, and keeps
  the SQLite database on the `act-data` volume (`/data`).
- With `AI_CONVERSATION_TREE_API_TOKEN` set, every API route requires
  `Authorization: Bearer <token>` — including `/ingest/conversations`. The
  viewer shell (`/ui`, `/assets`), `/healthz`, and `/` stay public so the page
  can load; the viewer asks for the token in *Settings* after its first `401`.
  The extension has a matching field. Compose refuses to start without a token.
- **Put TLS in front** (Caddy, nginx, a cloud load balancer). The token is a
  header; it must not cross the network in clear text.
- `AI_CONVERSATION_TREE_CORS_ORIGINS` adds browser origins (comma-separated)
  for a viewer served from a different domain; extension origins are always
  allowed.
- Reply generation: stub by default; set `OPENAI_API_KEY` for OpenAI. Provider
  keys entered in the viewer's Settings stay in that browser and are forwarded
  per request, as locally.
- `GET /healthz` backs the container's health check.

This is a single shared token on a single-user store: there are no accounts,
no per-user data isolation, and no rate limiting. The Dockerfile and compose
file are meant for one person's private server, not a multi-tenant service.

## Local Setup

### 1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Optional pgvector acceleration

For larger workspaces, start PostgreSQL with the included Compose file:

```bash
docker compose -f docker-compose.pgvector.yml up -d
```

That local connection is the default. Set `PGVECTOR_DATABASE_URL` when using a
different PostgreSQL instance. On the next request, the app creates the `vector` extension, mirrors new turn
embeddings into PostgreSQL, and uses an HNSW cosine index for older-turn
retrieval. SQLite remains the source of truth for conversations, turns, and
graph edges. If PostgreSQL is unavailable (connections time out after 3
seconds), retrieval automatically falls back to the local NumPy implementation.
Deleting a conversation, or rebuilding an imported one, removes its vectors from
the index, and retrieval ignores any vector for a turn the conversation's graph
doesn't have, so a stale index can't produce a link to a missing turn.

### 2. Frontend build

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Choose a default response mode

These environment variables set the *default* provider. Any conversation can
override it from the composer's model picker, `PATCH /conversations/{id}`, or a
`model` field on `POST .../turns`; `GET /models` lists what is available.

Resolution order when a conversation has no stored model: `OLLAMA_MODEL`, then
`AI_CONVERSATION_TREE_STUB_LLM=1`, then `OPENAI_API_KEY`.

#### Stub mode

```bash
export AI_CONVERSATION_TREE_STUB_LLM=1
```

#### Ollama mode

Example:

```bash
export OLLAMA_MODEL=qwen2.5:0.5b
export OLLAMA_BASE_URL=http://127.0.0.1:11434
unset AI_CONVERSATION_TREE_STUB_LLM
```

Use a small model for fast graph testing. The point is to generate turns quickly, not maximize answer quality.

#### OpenAI mode

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-5-mini
unset AI_CONVERSATION_TREE_STUB_LLM
```

### 4. Start the app

```bash
venv/bin/python -m uvicorn api:app --reload
```

Open:

- API: `http://127.0.0.1:8000`
- Viewer: `http://127.0.0.1:8000/ui` (served from `frontend/dist`)

For frontend development, run the backend on `:8000` and Vite separately:

```bash
cd frontend
npm run dev   # http://127.0.0.1:5173, proxies /conversations /edges /models /concepts to :8000
```

## Local Workflow

Typical workflow:

1. start a new chat — or capture existing ChatGPT / Claude / Gemini chats with
   the browser extension
2. send turns through the UI or API
3. open the graph drawer to see how the turns relate
4. hit `Reanalyze` after changing classifier logic

The viewer is for inspecting graph behavior; it is a real chat interface but a
local, single-user one, not a hosted product.

## Example Requests

### Create a conversation

```bash
curl -X POST http://127.0.0.1:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"title":"Test conversation"}'
```

### Add a turn

```bash
curl -X POST http://127.0.0.1:8000/conversations/1/turns \
  -H "Content-Type: application/json" \
  -d '{"userText":"What are cats?"}'
```

### Add a turn with a specific model

```bash
curl http://127.0.0.1:8000/models

curl -X POST http://127.0.0.1:8000/conversations/1/turns \
  -H "Content-Type: application/json" \
  -d '{"userText":"What are cats?","model":"ollama:qwen2.5:0.5b"}'
```

### Add a turn and watch it stream

```bash
curl -N -X POST http://127.0.0.1:8000/conversations/1/turns/stream \
  -H "Content-Type: application/json" \
  -d '{"userText":"What are cats?"}'
```

`-N` disables curl's output buffering so the `delta` events print as they
arrive instead of all at once at the end.

### Import a captured chat (what the extension sends)

```bash
curl -X POST http://127.0.0.1:8000/ingest/conversations \
  -H "Content-Type: application/json" \
  -d '{"source":"chatgpt","externalId":"abc123","title":"Cats",
       "messages":[{"role":"user","text":"What are cats?"},
                   {"role":"assistant","text":"Small carnivorous mammals."}]}'
```

Send it again with more messages and only the new turns are added. Add
`-H "Authorization: Bearer $TOKEN"` against a server that has a token set.

### Search, and check a chat's context health

```bash
curl "http://127.0.0.1:8000/search?query=index%20concurrently"
curl http://127.0.0.1:8000/conversations/1/context-health
```

### Get the graph

```bash
curl http://127.0.0.1:8000/conversations/1/graph
```

### Get the cross-conversation concept graph

```bash
curl http://127.0.0.1:8000/concepts/graph
curl http://127.0.0.1:8000/conversations/1/concept-links
```

### Reanalyze a conversation

```bash
curl -X POST http://127.0.0.1:8000/conversations/1/analyze
```

## Verification

Automated tests (pytest; no network, models load from the local Hugging Face
cache):

```bash
venv/bin/pip install -r requirements-dev.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python -m pytest
```

The suite covers ingestion (pairing, idempotent re-sync, append, rebuild,
validation), context-health scoring, hybrid search, summaries (including
provider failure), API-token auth and CORS (a 401 still carries CORS headers), and the conversation / turn / edge /
report / concept-link endpoints, plus regressions for atomic turn writes, cache
recovery after a failed write, concurrent first syncs, and stale pgvector rows.
Each test runs against a throwaway database.
The classifier is graded separately by the eval scripts below, and the extension
extractors by `npm test` in `extension/`.

Lint (unused imports / names):

```bash
venv/bin/pip install pyflakes && venv/bin/python -m pyflakes *.py scripts tests
```

Evaluation scripts:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python eval_immediate_previous.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python eval_cross_links.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python eval_concept_links.py
```

`eval_immediate_previous.py` and `eval_cross_links.py` cover in-conversation
classification; `eval_concept_links.py` builds small conversations in throwaway
databases and checks which ones end up cross-linked.

`eval_immediate_previous.py` reads its cases from `eval_cases.json` (see
"Immediate Previous Turn" above for the graded / `knownGap` split). If a
`knownGap` case starts passing, the runner prints `NOW PASSES` — promote it to
a graded case and drop the flag.

Frontend build:

```bash
cd frontend && npm run build
```

## Current Limitations

- when pgvector is not configured, older-turn retrieval uses an exact scan over
  the conversation's embedding matrix; the pgvector path uses HNSW ANN search
- concept linking still uses centroid shortlists in Python; the first pgvector
  integration covers turn retrieval, while a future pass can move workspace-wide
  concept linking into a materialized vector index too
- `same` vs `related` is a two-threshold heuristic; `all-MiniLM-L6-v2` cannot
  reliably separate genuinely adjacent topics from noise, so recall is
  conservative
- a manual concept link always has kind `same` or `related` at score `1.0`;
  there's no way to record a weaker hand-made connection
- local `Ollama` latency depends heavily on hardware and model size
- the graph cache is process-local but validates each entry against SQLite's
  `updatedAt` version; SQLite still remains the write coordination point
- Gemini replies currently fall back to one chunk; its streaming JSON protocol
  is not yet implemented
- auth is one shared token: no accounts, per-user isolation, or rate limiting
- the extension's DOM selectors need maintenance as the chat sites change, and
  they have been verified against fixture HTML, not the live sites
- context-loss detection is heuristic (it flags risk from repeated questions,
  "as I said" callbacks, and long-range dependencies); it does not read
  anything the model actually saw
- ingestion classifies on the request thread, so a very long first sync can
  take a minute or more

## Future Work

### Conversation Intelligence

Implemented: structured and model-written summaries, hybrid search with
turn-level navigation, Markdown/JSON report exports, and context-loss
detection.

Next:

- contextual signals the structure can't see: an assistant reply that
  contradicts an earlier answer (needs a model or NLI pass)
- summaries that cover each thread rather than the whole chat
- search filters by source and date

### Browser Extension

Implemented: capture for ChatGPT, Claude, and Gemini, idempotent incremental
sync, optional auto-sync, token auth.

Next:

- inject the graph as a side panel on the chat site itself, instead of the
  separate viewer tab
- Firefox build (the manifest is close; the service worker needs a background
  script shim)
- background (queued) ingestion so long first syncs return immediately
- live-site selector verification in CI

### Postgres + pgvector

The optional pgvector path now provides PostgreSQL-backed HNSW retrieval for
turn embeddings while SQLite remains the source of truth for graph metadata.
A future hosted migration could move the remaining graph tables to PostgreSQL
as well, and could use a materialized vector index for cross-conversation
concept scoring.

### Retrieval Improvements

When pgvector is disabled, older-turn retrieval loads prior embeddings and
scores them in Python (cosine minus a small age decay). The local fallback is
intentionally retained for easy offline development.

Planned improvements:

- better candidate pruning and age-decay tuning
- optional topic/subtopic segmentation for large conversations

### Model / Classifier Improvements

Planned work:

- add more labeled evaluation cases
- calibrate thresholds using saved examples
- improve continuation vs related separation
- improve branch detection for clarification subthreads
- add better handling for malformed or low-quality assistant replies
- make confidence scores more interpretable

### Concept Links

Planned work:

- filtering the workspace map to linked concepts only

### UI / Product Improvements

Planned work:

- streaming support for Gemini's `streamGenerateContent` protocol
- conversation rename (auto-titling from the first message is done)
- graph filtering by edge type and subgraph focus

## Notes

- Runtime data lives in `.data/conversationTree.db` by default. Set
  `AI_CONVERSATION_TREE_DB` to migrate or use another database path. A legacy
  root-level `conversationTree.db` is no longer used by default.
- If you use `Ollama`, make sure the local Ollama server is running before starting the backend.
- For quick local testing, use a smaller Ollama model rather than a larger chat model.
