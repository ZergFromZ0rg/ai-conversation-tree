# AI Conversation Tree

`AI Conversation Tree` is a local-first proof of concept for analyzing chat structure.

It takes a sequence of user/assistant turns, classifies how each new turn relates to earlier turns, stores the result, and renders the conversation both as a normal chat and as a graph.

Current goals:

- validate graph logic
- persist conversations locally
- test with local models through `Ollama`
- provide a chat-first viewer with a graph drawer for inspection

This is not intended to be production-ready. It is intended to be technically solid, explainable, and useful as a portfolio project.

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
- embeddings

and rendered as a chat with a graph drawer.

## Core Approach

### Immediate Previous Turn

The immediate previous turn gets the strongest classifier.

It uses three signal sources:

- embedding similarity
- discourse features
- cross-encoder scores

Those are combined to classify the relationship as:

- `continuation`
- `branch`
- `related`
- or no edge

### Older Prior Turns

Older-turn linking uses retrieval first, then classification.

Current flow:

1. score every earlier turn by cosine similarity to the new turn, minus a small
   decay for older turns
2. take the top few candidates
3. classify each selected older link as `continuation` or `related`

At this scale a direct scan is cheap and avoids the concept-centroid layer,
whose average embedding is a poor representative once a concept has drifted.

### Important Design Principle

Topic identity should dominate continuation decisions.

Discourse features are used as supporting evidence, not as ground truth. This reduces false positives where generic follow-up phrasing looks like continuation even when the topic changed.

## Tech Stack

Backend:

- `Python`
- `FastAPI`
- `SQLite` — `sqlite-vec` (vectors) via `apsw`, stdlib `sqlite3` for everything else
- `Pydantic`
- `NumPy`
- `sentence-transformers`
- `CrossEncoder`
- `PyTorch`
- `Ollama` for local response generation

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
- `graphStore.py`
  - per-conversation graph cache (LRU) and per-conversation write lock
- `db.py`
  - `SQLite` schema and helpers (`conversationTree.db`)
- `vectorStore.py`
  - `sqlite-vec` turn-embedding store (`conversationVectors.db`, via `apsw`)
- `models.py`
  - application data models

Frontend (`frontend/src/`):

- `App.jsx`
  - shell: conversation state, layout
- `api.js`
  - fetch wrappers
- `useConversation.js`
  - hook owning one conversation's turns + graph and the send / analyze / refresh actions
- `components/`
  - `ConversationSidebar`, `ChatTranscript`, `Composer`, `GraphRail`, `GraphDrawer`, `TurnGraph`

## Persistence Model

`conversationTree.db` (`SQLite`, WAL mode) stores:

- `conversations`
- `turns`
- `semanticEdges` — each edge carries an `origin` of `auto` (classifier) or `manual` (hand-created via `POST /edges`)
- `turnConcepts`

Turn embeddings live in a separate `conversationVectors.db`, in a `sqlite-vec`
`vec0` virtual table (native `float[384]` vectors, cosine metric). Similarity is
a `vec_distance_cosine(...)` query rather than JSON text parsed and looped over
in Python. It is accessed through `apsw` because the stdlib `sqlite3` build on
some platforms is compiled without loadable-extension support. That db file is
not committed; on startup any legacy `turnEmbeddings` JSON table found in
`conversationTree.db` is copied into it as a seed.

The first request for a conversation rebuilds its in-memory `ConversationGraph`
from persisted rows; `graphStore` then keeps it cached so subsequent turns do
not re-read the whole history. This assumes a single backend process (one
`uvicorn` worker).

## API

### Conversation Endpoints

- `POST /conversations`
- `GET /conversations`
- `GET /conversations/{conversationId}`
- `DELETE /conversations/{conversationId}`

### Turn Endpoints

- `POST /conversations/{conversationId}/turns`
- `GET /conversations/{conversationId}/turns`

### Graph Endpoints

- `POST /conversations/{conversationId}/analyze` — recomputes the classifier
  (`auto`) edges and every turn's root / concept assignments in a single
  transaction; `manual` edges are left in place
- `GET /conversations/{conversationId}/graph`

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

## Interface

The viewer at `/ui` is a normal chat: a conversation list on the left, the
transcript in the middle, a composer at the bottom.

The graph lives in a drawer on the right:

- a pill on the right edge shows the turn count; click it (or press
  `Cmd/Ctrl+G`) to slide the graph drawer open
- the graph is laid out top-to-bottom by `Dagre` along the primary-parent spine;
  branch and related links are drawn as secondary curves
- clicking a graph node scrolls the transcript to that turn and highlights it
- the drawer header has `Refresh` and `Reanalyze`

The UI follows the browser's light/dark preference. It does not stream replies
token by token yet.

## Local Setup

### 1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Frontend build

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Choose a response mode

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
npm run dev   # http://127.0.0.1:5173, proxies /conversations and /edges to :8000
```

## Local Workflow

Typical workflow:

1. start a new chat
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

### Get the graph

```bash
curl http://127.0.0.1:8000/conversations/1/graph
```

### Reanalyze a conversation

```bash
curl -X POST http://127.0.0.1:8000/conversations/1/analyze
```

## Verification

Backend syntax:

```bash
venv/bin/python -m py_compile api.py chatService.py graphService.py graphStore.py db.py vectorStore.py models.py
```

Persistence milestone:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python testSqliteMilestone.py
```

Classifier evaluation scripts:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python eval_immediate_previous.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python eval_cross_links.py
```

Frontend build:

```bash
cd frontend && npm run build
```

## Current Limitations

- older-turn retrieval scans every prior turn (`vec_distance_cosine` in SQL, but still one row per turn rather than an ANN index)
- local `Ollama` latency depends heavily on hardware and model size
- the in-memory graph cache assumes a single backend process (one `uvicorn` worker)
- replies are not streamed token by token
- no auth, multi-user isolation, or production deployment concerns are addressed

## Future Work

### Browser Extension

The most interesting product direction is a browser extension rather than a standalone app.

Goal:

- let users keep using `ChatGPT`, `Claude`, or `Gemini`
- read the visible conversation from the page
- build the conversation graph locally
- render the graph as a side panel

Planned work:

- browser extension scaffold
- content scripts for site-specific DOM extraction
- per-site adapters for `ChatGPT`, `Claude`, and `Gemini`
- local storage of extracted conversations
- graph viewer injected as a side drawer
- optional connection to the current local backend for graph construction

### Postgres + pgvector

`SQLite` + `sqlite-vec` is correct for a local proof of concept. A hosted
version would move to `Postgres` + `pgvector` for real concurrency, a proper
migration path, and an approximate-nearest-neighbour index (`sqlite-vec` KNN is
still a full scan today).

### Retrieval Improvements

Older-turn retrieval computes `vec_distance_cosine` for every prior turn in the
conversation (minus a small age decay). Fine for local conversations, but linear.

Planned improvements:

- an ANN index instead of a per-turn scan (see the `pgvector` note above)
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

### UI / Product Improvements

Planned work:

- stream assistant replies token by token
- auto-title conversations from the first message; conversation rename
- edge-correction tooling in the drawer (create / relabel / delete edges)
- graph filtering by edge type and subgraph focus

## Notes

- If `conversationTree.db` comes from an older schema version, delete it once and let the app recreate it.
- If you use `Ollama`, make sure the local Ollama server is running before starting the backend.
- For quick local testing, use a smaller Ollama model rather than a larger chat model.
