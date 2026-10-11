# Using It As A Library

The logic layer is a plain Python package, `conversation_tree`, that another application can import: configure it once, record turns, plug in your own model layer.

[← Back to the README](../README.md)

The server and the viewer are one way to use this. The logic layer is a plain
Python package, `conversation_tree`, that another application can import. It
needs only NumPy, `sentence-transformers`, and `httpx`:

```bash
pip install -e .             # core logic only
pip install -e ".[server]"   # + the FastAPI layer and uvicorn
pip install -e ".[pgvector]" # + PostgreSQL/pgvector retrieval
```

## Configuration

Importing the package reads no environment, opens no database, and loads no
model. The host configures it once:

```python
import conversation_tree
from conversation_tree import Settings

conversation_tree.init(Settings(dbPath="/var/lib/myapp/conversations.db"))
conversation_tree.warmup()   # optional: load the embedding models now, not on first use
```

`Settings` checks itself when it is created: `dbPath` is made absolute (and `~`
expanded) at that moment, so a host that changes directory later keeps its
database, and `contextWindowTokens` must be a positive integer. `init()` creates
the database (idempotent) and drops any state tied to a previous configuration: the cached graphs, the concept-link timing, cached
model-written summaries, and the pgvector schema state. It is all-or-nothing:
if the settings or the replier are invalid, or the database cannot be opened, it
raises and the package keeps running exactly as it was configured before. Services called before it raise `NotConfiguredError`.
`init()` with no argument reads the environment, which is what the bundled app
does:

| `Settings` field | Environment variable | Default from the environment |
| --- | --- | --- |
| `dbPath` | `AI_CONVERSATION_TREE_DB` | `.data/conversationTree.db` under the working directory |
| `contextWindowTokens` | `AI_CONVERSATION_TREE_CONTEXT_TOKENS` | `8000` (also the default of a plain `Settings`); measures context loss and caps the history sent with each message |
| `pgvectorUrl` | `PGVECTOR_DATABASE_URL`, then `DATABASE_URL` | the local Compose database (a plain `Settings` leaves pgvector off) |

The HTTP-only options (`AI_CONVERSATION_TREE_API_TOKEN`,
`AI_CONVERSATION_TREE_CORS_ORIGINS`) live in `api.ServerOptions`, so the logic
layer never sees them. `createApp(settings, options, replier)` configures the
package in its startup hook and loads the models before accepting requests;
serve it with `uvicorn --factory conversation_tree.api:createApp`. The built-in
reply providers (`OPENAI_API_KEY`, `OLLAMA_BASE_URL`,
`AI_CONVERSATION_TREE_STUB_LLM`, …) read their own settings from the environment
when a reply is generated; a host that plugs in its own replier (below) never
touches them.

The embedding models (`all-MiniLM-L6-v2` and its cross-encoder) are fixed, not
settings: the classifier thresholds were tuned for them and the pgvector table is
sized for their 384 dimensions.

## Where Things Live

Import from the modules directly (`from conversation_tree.chatService import
appendTurn`); the HTTP layer is not needed.

| To do this | Call |
| --- | --- |
| create, read, list, and delete conversations; set a conversation's model | `chatService.createConversationSession`, `getConversationSession`, `listConversationSessions`, `deleteConversationSession`, `setConversationSessionModel` |
| record a finished exchange | `chatService.appendTurn` |
| generate a reply and record it | `chatService.processChatMessage`, `streamChatMessage` |
| import a whole chat (idempotent) | `ingestService.ingestConversation` |
| read turns and the graph; re-run the classifier | `chatService.listConversationTurns`, `listConversationGraph`, `analyzeConversation` |
| correct edges and concept links by hand | `chatService.createConversationEdge`, `patchConversationEdge`, `removeConversationEdge`, `createManualConceptLink`, `removeConceptLink` |
| cross-conversation concepts | `chatService.listConceptGraph`, `listConversationConceptLinks`; `conceptIndex.relinkAllConceptLinks` |
| search, summarise, export | `analysisService.searchConversations`, `buildConversationSummary`, `buildLlmSummary`, `buildConversationReport`, `reportAsMarkdown` |
| check for context loss | `contextService.buildContextHealth` |

## Recording Turns You Generated Yourself

A host that already has the reply (its own chat, its own model call) hands over
the finished exchange; nothing is generated and no provider is involved:

```python
from conversation_tree.chatService import appendTurn, createConversationSession

conversationId = createConversationSession()["conversationId"]
result = appendTurn(conversationId, "What are cats?", "Small carnivorous mammals.")
result["turnId"], result["edges"]   # the turn is classified, persisted, and linked
```

`appendTurn` classifies the turn against the conversation's earlier ones,
persists it, titles the conversation from its first question, and refreshes
cross-chat concept links, exactly as a turn typed into the viewer would.

## Plugging In Your Own Model Layer

`processChatMessage`, `streamChatMessage`, and the model-written summary ask a
*replier* for text. Pass a plain function of a `ReplyRequest`, or subclass
`providers.Replier` for streaming, model validation, and a default model:

```python
conversation_tree.init(settings, replier=lambda request: myLlm.chat(request.history, request.userText))

class MyReplier(providers.Replier):
    modelSpecHint = "model must be one of my models."
    def generate(self, request): ...
    def stream(self, request): ...            # optional
    def isValidModelSpec(self, modelSpec): ...  # optional
    def defaultModelSpec(self): ...           # optional
    def contextWindowTokens(self, modelSpec): ...  # optional: this model's window, or None
```

A `ReplyRequest` carries `userText` (the new message), `modelSpec`, `apiKey`,
`conversationId`, and `history`. It is one object so a field can be added later
without changing every replier. `history` is the conversation so far, oldest
first, as `{"role": "user" | "assistant", "content": "..."}`. Use it as given,
or ignore it and look the conversation up by `conversationId` in your own store.
Turns recorded with `appendTurn` count as history like any other.

Only whole turns that fit the model's context window are included, newest
first, after the new message has been charged against it (four characters are
estimated as one token). The window is `Settings.contextWindowTokens` unless the
replier reports one for that model with `contextWindowTokens(modelSpec)`; a
replier serving models with different windows should, and a value that is not a
positive integer or `None` is an error rather than a silent fallback. That is the same window the context
health panel draws its horizon from (for the conversation's own model), so a turn the panel marks as outside the
window is also a turn the model is not sent. The first turn has an empty
history. A summary request has none either, and `conversationId` is `None`,
because its prompt already contains the transcript.

An exception from a plain function is reported as `ProviderError`; a subclass
should raise `ProviderError` itself for failures the caller ought to see as a
provider problem, and anything else propagates as a bug. A reply must be real
text: an empty, whitespace-only, or non-string reply (or stream chunk) is also a
`ProviderError`, and nothing is recorded. Pass a replier *instance* (`MyReplier()`),
not the class; the class is refused up front. An explicit model on a request becomes
the conversation's default only after that turn has succeeded. With no replier the
built-in providers are used: they send `history` in each provider's own format
(Ollama, OpenAI, Anthropic, and Gemini; the stub ignores it), so replies from a
real model see the earlier conversation.

## Errors And Validation

The HTTP layer is a thin skin. Every rule (label and confidence ranges, edge
direction, model-spec format, search limits, import sources) is enforced inside
the service functions, so a caller that is not FastAPI gets the same behaviour.
Failures are signalled with the exceptions in `conversationErrors.py`, never
with `None` / `False` returns or transport errors:

| Exception | Meaning | HTTP | Also a |
| --- | --- | --- | --- |
| `NotFoundError` | the conversation, edge, concept, or link doesn't exist | 404 | `LookupError` |
| `InvalidInputError` | the request breaks a domain rule | 422 | `ValueError` |
| `ProviderError` | the model provider failed, is unreachable, or isn't configured | 502 | `RuntimeError` |
| `ConflictError` | a write collides with an existing record (a duplicate import); `ingestConversation` handles it itself, so no endpoint returns it | | |
| `NotConfiguredError` | a service ran before `init()` | | `RuntimeError` |

All derive from `ConversationTreeError`. Provider calls translate `httpx`
errors into `ProviderError`, so callers never depend on the HTTP client.
`streamChatMessage` validates when it is *called* and raises before returning
its iterator, so a caller can still reject a bad request before streaming;
`ProviderError` surfaces while iterating.

What stays in `api.py` on purpose: request-shape limits (field lengths, message
counts) on the Pydantic models, the report's `format` and download filename, and
the auth/CORS middleware.

## Things To Know

- **Everything is synchronous.** Classifying a turn takes a second or two on
  CPU, so an async host should call these functions from a worker thread
  (`asyncio.to_thread`). The first classification loads two models; call
  `warmup()` at start-up to pay that cost early.
- **One configuration per process.** `init()` sets module-level state, so a
  process works against one database at a time. Calling it again switches
  databases and clears the caches; it is meant for start-up and tests.
- **SQLite is the store.** `db.py` is the only module that imports `sqlite3`
  (`vectorStore.py` reads embeddings through it for the optional pgvector
  mirror). There is no storage interface yet; see [Future Work](development.md#future-work).
- **History is the timeline.** A reply is sent the conversation in the order it
  happened, not just the semantic thread the new message belongs to.
