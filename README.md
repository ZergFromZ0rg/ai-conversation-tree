# AI Conversation Tree

`AI Conversation Tree` is a local full-stack proof-of-concept that turns a chat into a semantic graph.

Instead of storing a conversation as a flat list of messages, this project classifies how each new turn relates to earlier turns and visualizes those relationships as a tree/graph. It combines:

- embeddings for semantic similarity
- discourse-feature heuristics for conversational intent
- cross-encoder scoring for label-specific relevance
- `SQLite` persistence for saved conversations
- a `FastAPI` backend and browser UI for chat + graph exploration

This project is meant to be a strong local demo and portfolio piece rather than a production system.

## What It Does

When a user sends a message, the backend:

1. generates an assistant response
2. embeds the new turn
3. classifies its relationship to the immediately previous turn
4. retrieves older relevant turns by concept centroid similarity
5. assigns semantic edges such as `branch`, `continuation`, or `related`
6. saves the turn, embeddings, edges, and concept ids
7. returns updated conversation and graph data to the UI

The result is a conversation that can be explored both as:

- a normal saved chat
- a graph of semantic relationships across turns

## Why This Project Is Interesting

This repo demonstrates:

- LLM orchestration
- embedding-based retrieval
- discourse-aware classification
- cross-encoder reranking / scoring
- `SQLite` data modeling
- API design with `FastAPI`
- local full-stack product thinking
- graph-based visualization of conversational structure

## Current Architecture

### Backend

- `api.py`
  - HTTP routes
- `chatService.py`
  - orchestration flow for chat, LLM calls, persistence, and response payloads
- `graphService.py`
  - semantic classification and graph-building logic
- `db.py`
  - `SQLite` schema and persistence helpers
- `models.py`
  - app-level data models

### Storage

`SQLite` is the source of truth for:

- conversations
- turns
- semantic edges
- concept ids
- turn embeddings

The app currently rebuilds in-memory graph state from `SQLite` when a conversation is loaded. That keeps the current graph logic simple while still providing persistence.

### Frontend

- `frontend/`
  - Phase 1 React frontend
  - chat-first layout with:
    - center linear chat
    - left conversation drawer
    - right graph drawer
- `extension/`
  - placeholder plan for browser integration

## Classifier Overview

There are two main classification paths.

### 1. Immediate Previous Turn

The immediate previous turn gets the strongest classifier:

- embedding similarity
- discourse feature extraction
- label-specific cross-encoder scoring
- final label selection

Possible labels:

- `branch`
- `continuation`
- `related`
- effectively `unrelated` when no label is strong enough

### 2. Older Prior Turns

Older turns are handled with retrieval first:

- score concept centroids
- select top concepts
- score only turns inside those concepts
- classify each older link as `continuation` or `related`

This avoids scanning the entire conversation history on every message.

## Persistence Model

The current schema in `db.py` includes:

- `conversations`
- `turns`
- `semanticEdges`
- `turnConcepts`
- `turnEmbeddings`

Embeddings are stored as JSON text for simplicity in this proof-of-concept.

## Local Run

### 1. Install dependencies

Create and activate a virtual environment, then install the requirements.

Example:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Choose response mode

#### Stub mode

This is the easiest local demo path and does not require an API key.

```bash
export AI_CONVERSATION_TREE_STUB_LLM=1
```

#### Real OpenAI mode

This uses the OpenAI Responses API from `chatService.py`.

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-5-mini
unset AI_CONVERSATION_TREE_STUB_LLM
```

### 3. Start the app

```bash
venv/bin/python -m uvicorn api:app --reload
```

### 4. Open the UI

#### React UI

The new React frontend lives in `frontend/`.

Start the backend:

```bash
venv/bin/python -m uvicorn api:app --reload
```

Then in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The Vite dev server proxies API calls to the FastAPI backend on `http://127.0.0.1:8000`.

## Local Milestone Test

This repo includes a small milestone script that checks the first persistence flow:

- create a conversation
- send one message
- generate/store one turn
- reload it from `SQLite`

Run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python testSqliteMilestone.py
```

## Useful Endpoints

- `POST /conversations`
- `GET /conversations`
- `GET /conversations/{conversationId}`
- `POST /chat`
- `POST /debug/immediate`
- `GET /graph`

### Example `POST /chat`

```json
{
  "conversationId": 1,
  "userText": "What is Python?"
}
```

## Notes

- If you created `conversationTree.db` before the latest schema changes, delete it once and let the app recreate it.
- This project is designed for local use and portfolio/demo value, not production deployment.
- The current UI is now served from the built React frontend in `frontend/dist`.

## What Exists Right Now

Current implemented pieces:

- semantic graph-building backend
- immediate-turn and older-turn classification
- `SQLite` persistence
- saved conversation loading
- conversation history sidebar in the UI
- local browser graph visualization
- stub or real OpenAI response generation path

## Frontend Folder Structure

### `frontend/`

Phase 1 React app:

- `frontend/src/app.tsx`
  - app shell and state management
- `frontend/src/components/conversationSidebar.tsx`
  - saved chat history drawer
- `frontend/src/components/chatThread.tsx`
  - linear conversation view
- `frontend/src/components/chatComposer.tsx`
  - message composer
- `frontend/src/components/graphDrawer.tsx`
  - right-side graph drawer
- `frontend/src/lib/api.ts`
  - typed client for FastAPI endpoints
- `frontend/src/types.ts`
  - shared frontend types
- `frontend/src/styles.css`
  - app styling

### `extension/`

Future browser integration path:

- browser extension shell
- content scripts for ChatGPT / Claude / Gemini
- site-specific turn extraction adapters
- browser-local storage, likely `IndexedDB`
- optional localhost connection back to this Python backend for graph construction

## How the Future Extension Should Work

The longer-term browser-companion version should work like this:

1. user chats on an existing AI site
2. extension reads visible chat turns from the page
3. extension sends those turns to the local backend
4. backend runs embeddings, classification, and graph building
5. extension renders the graph in a side panel

That preserves the current Python graph logic while avoiding extra response-generation API costs inside this project.
- evaluation scripts for the classifier logic

## Next Steps / Future Improvements

### Phase 1: Product Improvements

- Add a fuller chat history panel beside the graph, not just the conversation list and graph view.
- Improve the visual design so the tool feels more intentional and polished for demo use.
- Add message/node linking so selecting a graph node highlights the corresponding turn in the chat view.
- Add screenshots or a short demo GIF for GitHub and LinkedIn.

### Phase 1: Retrieval Improvements

- The current older-turn retrieval path uses concept centroids to avoid scanning the full conversation history, then scores only turns inside the top matching concepts.
- A future optimization is to index turns within each concept as well, so once the best concepts are chosen the system can retrieve the top turn candidates directly instead of scanning every turn in those concepts.

### Phase 2: Production-Style Storage

- Migrate from `SQLite` to `Postgres + pgvector` when stronger concurrency and production reliability are needed.
- Store embeddings in the database and use vector similarity search for semantic retrieval.
- Move older-turn retrieval and context selection closer to database-level nearest-neighbor queries.
- Support multi-user conversations and more scalable chat history management.

### Phase 2: ML / System Improvements

- Improve evaluation coverage with larger labeled datasets for immediate-turn and older-turn classification.
- Add calibrated confidence scoring and richer offline evaluation.
- Improve retrieval, summarization, and semantic memory for long conversations.
- Add observability for model calls, retrieval quality, and graph behavior.

## Project Positioning

This is best presented as:

**An AI conversation-graph prototype that uses embeddings, discourse features, and cross-encoder scoring to classify conversational structure, persist chat state, and visualize semantic relationships across turns.**
