# Development

How the code is laid out, how it is tested, what it does not do yet, and what comes next.

[← Back to the README](../README.md)

## Tech Stack

Core logic (`pip install .`, Python 3.10+):

- `NumPy`
- `sentence-transformers` and a `CrossEncoder`, on `PyTorch`
- `SQLite`
- `httpx`, for the model providers

Optional:

- `FastAPI`, `Pydantic`, and `uvicorn` — the HTTP layer and viewer shell
  (`.[server]`)
- `psycopg` + `pgvector` — scalable turn retrieval (`.[pgvector]`)
- `Ollama` (local), `OpenAI`, `Anthropic`, or `Gemini` for response generation;
  a stub mode needs none of them

Frontend:

- `React` + `Vite`
- `React Flow` for the graph canvas
- `Dagre` for hierarchical graph layout

## Project Structure

Backend (the `conversation_tree/` package; importing the package itself is
cheap, the embedding models load when `graphService` is first imported):

- `api.py`
  - HTTP routes, auth, CORS, and the viewer shell, built by `createApp()`.
    Holds no domain rules: it maps the services' exceptions to status codes.
    Needs the `server` extra
- `config.py`, `lifecycle.py`
  - `Settings`, and `init()` / `warmup()` (see [Configuration](library.md#configuration))
- `__main__.py`
  - `python -m conversation_tree`: a tiny REPL that classifies turns you type
    in, no server or database involved
- `conversationErrors.py`
  - the services' error contract (see [Errors And Validation](library.md#errors-and-validation))
- `chatService.py`
  - orchestration (conversations, turns, edges, concept links) and graph payload
    serialization. `appendTurn` is the entry point for recording a finished exchange
- `providers.py`
  - where replies come from: the `Replier` a host can plug in, and the built-in
    Ollama / OpenAI / Anthropic / Gemini / stub routing behind it
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
  - turns imported role-tagged messages into turns, matches a chat to an
    earlier import, and classifies only what is new
- `contextService.py`
  - context-loss detection (pure scoring function plus a DB wrapper), and
    `recentHistory`, which picks the earlier turns that fit the context window
- `db.py`
  - `SQLite` schema and helpers
- `models.py`
  - the `TurnModel` dataclass

Elsewhere:

- `pyproject.toml` — package metadata; dependencies split into the core logic
  and the `server`, `pgvector`, and `dev` extras
- `requirements.txt` — the tested, pinned set that the Docker image and the
  quick start install
- `tests/` — the pytest suite
- `eval_*.py`, `eval_cases.json` — accuracy harnesses for the classifier and the
  concept links (see [Verification](#verification))
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

## Verification

Automated tests (pytest; no network, models load from the local Hugging Face
cache):

```bash
venv/bin/pip install -r requirements-dev.txt   # or: pip install -e ".[dev]"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/pytest
```

`pytest` reads its settings (including the import path, so no install is needed)
from `pyproject.toml`, which must stay the only pytest config file.

The suite covers ingestion (pairing, idempotent re-sync, append, rebuild,
validation), context-health scoring, hybrid search, summaries (including
provider failure), API-token auth and CORS (a 401 still carries CORS headers), and the conversation / turn / edge /
report / concept-link endpoints, plus regressions for atomic turn writes, cache
recovery after a failed write, concurrent first syncs, and stale pgvector rows.
For the library side it also checks, with no HTTP involved, that every service
enforces its own rules and raises the documented exception; that settings,
`init()`, and the lazy models behave (including a subprocess check that
importing the package reads no environment, opens no database, and loads no
model); `appendTurn`; the replier plug-in; which earlier turns a replier
receives and how the context window trims them; and the exact request each
built-in provider sends. Each test runs against a throwaway database.
The classifier is graded separately by the eval scripts below.

Lint (unused imports / names):

```bash
venv/bin/pip install pyflakes && venv/bin/python -m pyflakes conversation_tree *.py scripts tests
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

`eval_immediate_previous.py` reads its cases from `eval_cases.json` (see [Immediate Previous Turn](how-it-works.md#immediate-previous-turn) above for the graded / `knownGap` split). If a
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
- context-loss detection is heuristic (it flags risk from repeated questions,
  "as I said" callbacks, and long-range dependencies); it does not read
  anything the model actually saw
- ingestion classifies on the request thread, so a very long first sync can
  take a minute or more
- the services are synchronous, and the package holds one configuration per
  process (see [Things To Know](library.md#things-to-know))
- SQLite is the only store, and there is no storage interface; the embedding
  models are fixed rather than configurable
- the history sent to a model is the whole timeline trimmed by a rough
  characters-per-token estimate, not the semantic thread, and the exact token
  count of a given model will differ; the built-in providers do not know any
  model's window, so lower `contextWindowTokens` for the smallest model you use
  or report windows from your own replier
- the history payloads for OpenAI, Anthropic, and Gemini follow each provider's
  documented format and are tested against the request shapes, but have not been
  exercised against the live APIs

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

### Host Integration

Next:

- a storage interface in front of `db.py`, so a host can keep conversations in
  its own database
- one public module that re-exports the host-facing functions, so callers need
  not know which service each lives in
- an injectable embedding model (and a matching pgvector dimension), instead of
  the fixed MiniLM pair
- thread-aware history, sending the semantic thread a message belongs to rather
  than the whole timeline

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

- Runtime data lives in `.data/conversationTree.db` (relative to the working
  directory) by default. Set
  `AI_CONVERSATION_TREE_DB` to migrate or use another database path. A legacy
  root-level `conversationTree.db` is no longer used by default.
- If you use `Ollama`, make sure the local Ollama server is running before starting the backend.
- For quick local testing, use a smaller Ollama model rather than a larger chat model.
