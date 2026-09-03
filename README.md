# AI Conversation Tree

`AI Conversation Tree` is a local-first proof of concept for analyzing chat structure.

It takes a sequence of user/assistant turns, classifies how each new turn relates to earlier turns, stores the result, and renders the conversation as a graph.

Current goals:

- validate graph logic
- persist conversations locally
- test with local models through `Ollama`
- provide a simple graph viewer for inspection

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

and rendered as a graph.

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

1. compare the new turn to concept centroids
2. select the strongest concepts
3. score turns inside those concepts
4. classify selected older links as `continuation` or `related`

This avoids a full scan over the entire conversation history on every turn.

### Important Design Principle

Topic identity should dominate continuation decisions.

Discourse features are used as supporting evidence, not as ground truth. This reduces false positives where generic follow-up phrasing looks like continuation even when the topic changed.

## Tech Stack

- `Python`
- `FastAPI`
- `SQLite`
- `Pydantic`
- `NumPy`
- `sentence-transformers`
- `CrossEncoder`
- `PyTorch`
- `React Flow`
- `Ollama` for local response generation

## Project Structure

- `api.py`
  - HTTP routes
- `chatService.py`
  - orchestration, provider calls, graph payload serialization
- `graphService.py`
  - classification logic, concept retrieval, reanalysis
- `db.py`
  - `SQLite` schema and helpers
- `models.py`
  - application data models
- `frontend/`
  - minimal graph viewer

## Persistence Model

`SQLite` stores:

- `conversations`
- `turns`
- `semanticEdges`
- `turnConcepts`
- `turnEmbeddings`

Embeddings are currently stored as JSON for simplicity.

When a conversation is loaded, the backend rebuilds the in-memory graph state from persisted rows.

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

- `POST /conversations/{conversationId}/analyze`
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
- Viewer: `http://127.0.0.1:8000/ui`

## Local Workflow

Typical workflow:

1. create a conversation
2. send turns through the UI or API
3. inspect the graph
4. re-run `Analyze` after logic changes

The UI is intentionally minimal. It is for inspecting graph behavior, not for polished chat UX.

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
python3 -m py_compile api.py chatService.py graphService.py db.py models.py
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

## Current Limitations

- older-turn retrieval still depends on centroid-first heuristics
- embeddings are stored as JSON, not a native vector type
- local `Ollama` latency depends heavily on hardware and model size
- UI is an inspection tool, not a complete chat product
- graph layout is heuristic, not a full semantic DAG layout engine
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

The current `SQLite` backend is correct for a local proof of concept.

The next backend upgrade would be:

- `Postgres`
- `pgvector`

Benefits:

- embeddings stored in a native vector column
- database-side nearest-neighbor search
- better concurrency and migration path
- more realistic ML backend architecture

This would replace manual embedding retrieval in Python with database-backed vector search.

### Retrieval Improvements

The current older-turn retrieval uses:

- concept centroids first
- then turn scoring inside the chosen concepts

That is already better than scanning every turn, but it is still approximate.

Planned improvements:

- maintain a per-concept turn index instead of rescoring all turns in selected concepts
- refine centroid updates and concept membership logic
- add stronger topic clustering or subtopic segmentation
- improve candidate pruning and age decay
- eventually replace heuristic retrieval with vector search over persisted embeddings

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

- better graph legend and filtering
- node focus / centering and subgraph inspection
- cleaner tree layout for dense graphs
- chat-first viewer with graph drawer
- correction tooling for manually editing edges

## Notes

- If `conversationTree.db` comes from an older schema version, delete it once and let the app recreate it.
- If you use `Ollama`, make sure the local Ollama server is running before starting the backend.
- For quick local testing, use a smaller Ollama model rather than a larger chat model.
