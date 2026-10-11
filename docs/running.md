# Running It

Local setup, choosing how replies are generated, optional pgvector, and hosting it on a private server.

[← Back to the README](../README.md)

## Local Setup

### 1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To use the logic as a library in another project instead, see [Using It As A Library](library.md).

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

#### Anthropic and Gemini

These are chosen per conversation rather than by default: pick or send an
`anthropic:<model>` or `gemini:<model>` spec, and provide `ANTHROPIC_API_KEY` /
`GEMINI_API_KEY` (or a key entered in the viewer's Settings, which is forwarded
per request).

A replier passed to `init(..., replier=...)` replaces all of the above.

### 4. Start the app

```bash
venv/bin/python -m uvicorn --factory conversation_tree.api:createApp --reload
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

1. start a new chat — or import an existing one through `/ingest/conversations`
2. send turns through the UI or API
3. open the graph drawer to see how the turns relate
4. hit `Reanalyze` after changing classifier logic

The viewer is for inspecting graph behavior; it is a real chat interface but a
local, single-user one, not a hosted product.

## Hosted Deployment

For an always-on server (so chats can be imported and viewed from any machine):

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
  Compose refuses to start without a token.
- **Put TLS in front** (Caddy, nginx, a cloud load balancer). The token is a
  header; it must not cross the network in clear text.
- `AI_CONVERSATION_TREE_CORS_ORIGINS` adds browser origins (comma-separated)
  for a viewer served from a different domain; the Vite dev server origins
  are always allowed.
- Reply generation: stub by default; set `OPENAI_API_KEY` for OpenAI. Provider
  keys entered in the viewer's Settings stay in that browser and are forwarded
  per request, as locally.
- `GET /healthz` backs the container's health check.

This is a single shared token on a single-user store: there are no accounts,
no per-user data isolation, and no rate limiting. The Dockerfile and compose
file are meant for one person's private server, not a multi-tenant service.
