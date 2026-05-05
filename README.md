# ai-conversation-tree
ai-conversation-tree

## Next Steps / Future Improvements

### Phase 1: Solid V1 Backend

- Add `SQLite` persistence for conversations, turns, embeddings, semantic edges, and concept ids.
- Save and load chat history instead of keeping all state only in memory.
- Add a `POST /chat` backend flow that:
  - receives the user message
  - calls an LLM to generate `aiText`
  - runs graph/classification logic
  - saves the turn, embedding, semantic edges, and concept ids
  - returns updated conversation and graph data
- Separate responsibilities into:
  - `api.py`
  - `chatService.py`
  - `graphService.py`
  - `db.py`
  - `models.py`

### Phase 1: Frontend / Product

- Build a simple chat UI for sending messages and viewing turn history.
- Add a graph visualization beside the chat so users can inspect conversation structure.
- Add a conversation list / history view so saved chats can be reopened.
- Link chat messages and graph nodes so selecting one highlights the other.

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
