# HTTP API

Every endpoint of the optional HTTP layer, the graph JSON it returns, importing existing chats, and `curl` examples.

[← Back to the README](../README.md)

## Endpoints

### Model Endpoints

- `GET /models` — installed `Ollama` tags (queried live from `/api/tags`), the
  configured `OpenAI` model, and `stub`, plus the current default and whether
  `Ollama` is reachable
- `GET /ollama/library` — a short curated list of Ollama models that are not
  installed yet, and whether Ollama is reachable
- `POST /ollama/pull` — `{"model": "<name>"}`; streams Ollama's pull progress as
  `text/event-stream`. A failure arrives as a
  `{"status": "error", "message": "..."}` event, not an HTTP error

### Conversation Endpoints

- `POST /conversations` — optional `model` sets the conversation default
- `GET /conversations` — each entry carries `source` / `sourceUrl` when the
  chat was imported (null otherwise)
- `GET /conversations/{conversationId}`
- `PATCH /conversations/{conversationId}` — set the conversation's default
  response model (`{"model": "ollama:<name>"}`); validated by the installed
  replier (the built-in one accepts `stub` and `ollama:` / `openai:` /
  `anthropic:` / `gemini:` specs)
- `DELETE /conversations/{conversationId}`

### Turn Endpoints

- `POST /conversations/{conversationId}/turns` — optional `model` overrides the
  provider for this turn and becomes the conversation's new default; the model is
  sent the conversation so far (the earlier turns that fit the context window),
  not just the new message; a provider failure returns `502`. If this is the conversation's first turn and it has no
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
  with `200`, so there's no status left to change. A missing conversation or an
  invalid model is still a real `404` / `422`, because those are checked before
  streaming starts. `Ollama`, `OpenAI`, `Anthropic`, and the stub stream
  incremental chunks; Gemini currently sends its whole reply as one `delta`
- `GET /conversations/{conversationId}/turns`

### Ingest Endpoint

- `POST /ingest/conversations` — create or update a conversation imported
  from elsewhere. Body: `{"source": "chatgpt"|"claude"|"gemini"|"other",
  "externalId": "...", "title"?, "url"?, "messages": [{"role": "user"|"assistant",
  "text": "..."}]}`. See [Importing Conversations](#importing-conversations) for the exact semantics. Returns
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
  (see [Context-Loss Detection](how-it-works.md#context-loss-detection))
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

### Viewer And Health Endpoints

- `GET /` — an index of the endpoints (public)
- `GET /ui` — the built viewer; `503` with the build steps if `frontend/dist`
  does not exist
- `GET /healthz` — liveness check (public)

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

## Importing Conversations

`POST /ingest/conversations` takes a chat that already exists somewhere else (a
data export, another app, a script) as a flat list of role-tagged messages and
runs it through the same pipeline as a typed chat. `scripts/seed_demo.py` is a
working example client.

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
  A rebuild is all-or-nothing: the new history is classified first, off to the
  side, and then swapped in by one transaction, so a failure part-way leaves the
  previous history untouched. Appending new turns saves them one at a time, so a
  failure there keeps the turns already saved and the next sync continues.
- Imported turns go through the same classifier, concept assignment, and
  cross-chat linking as typed ones, so they show up in the graph, the map,
  search, and context-health immediately. Classifying on CPU takes roughly a
  second or two per turn.

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

### Import an existing chat

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
