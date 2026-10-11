# AI Conversation Tree

Turn a chat transcript into a map you can search and audit. It works out how each
turn relates to the ones before it, links topics that recur across your
conversations, and warns you when a long chat has probably lost its own context.

![The conversation graph: a seven-turn chat with its turns linked by continuation and related edges, a context-loss warning in the transcript, and an "also discussed elsewhere" panel.](docs/images/conversation-graph.png)

- **A graph of every chat.** Each turn is a continuation, a branch, related to an
  earlier turn, or a fresh topic, with a confidence score you can correct by hand.
- **Links across chats.** "You also talked about this in another conversation."
- **Search** by meaning and by exact words, straight to the turn.
- **Context-loss warnings.** Repeated questions, "as I said…", and the point where
  earlier messages fall out of the model's window.
- **Local-first.** SQLite and two small embedding models. No accounts, and no API
  key needed.

Run it as a small **server with a viewer**, or import it as a **Python library**
into your own app. It is built for one person's machine or private server (there
are no accounts and no per-user isolation), and it is a portfolio project rather
than a hosted product.

## Quick start

```bash
git clone https://github.com/ZergFromZ0rg/ai-conversation-tree.git
cd ai-conversation-tree
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# start the server and viewer (canned replies, no API key; the small context
# window just makes the demo's "outside the window" marker easy to see)
AI_CONVERSATION_TREE_CONTEXT_TOKENS=400 python -m uvicorn --factory conversation_tree.api:createApp
```

In a second terminal, load a few demo chats, then open <http://127.0.0.1:8000/ui>:

```bash
source venv/bin/activate && python scripts/seed_demo.py
```

The first start downloads the two embedding models (about 180 MB). The viewer is
prebuilt, so there is nothing to compile. Press `Cmd/Ctrl+G` for the graph and
`Cmd/Ctrl+K` to search. [A guided tour of the viewer →](docs/viewer.md)

## Use it as a library

```bash
pip install -e .          # just the logic: numpy, sentence-transformers, httpx
```

```python
import conversation_tree
from conversation_tree import Settings
from conversation_tree.chatService import appendTurn, createConversationSession

conversation_tree.init(Settings(dbPath="conversations.db"))

conversationId = createConversationSession()["conversationId"]
appendTurn(conversationId, "What are cats?", "Small carnivorous mammals.")
result = appendTurn(conversationId, "What do they eat?", "Mostly meat.")
result["edges"]   # how the new turn relates to the earlier ones
```

Importing the package does nothing: no database, no model, no environment. You
call `init()` once. To let it write the replies too, pass your own model layer,
which is handed the conversation so far:

```python
conversation_tree.init(settings, replier=lambda request: myModel(request.history, request.userText))
```

[Everything the library offers →](docs/library.md)

## Configuration

The server reads a few environment variables; all are optional.

| Variable | Default | What it does |
| --- | --- | --- |
| `AI_CONVERSATION_TREE_DB` | `.data/conversationTree.db` | where conversations are stored |
| `AI_CONVERSATION_TREE_STUB_LLM` | `1` | `1` answers with canned replies; unset it to use a real model |
| `OLLAMA_MODEL` | | reply with a local [Ollama](https://ollama.com) model |
| `OPENAI_API_KEY` | | reply with OpenAI (Anthropic and Gemini are chosen per chat) |
| `AI_CONVERSATION_TREE_CONTEXT_TOKENS` | `8000` | the context window assumed for warnings and for the history sent to a model |
| `AI_CONVERSATION_TREE_API_TOKEN` | | require `Authorization: Bearer …`; set this on any shared server |

## Learn more

| | |
| --- | --- |
| [Using it as a library](docs/library.md) | configuration, recording turns, plugging in your model, errors |
| [How it works](docs/how-it-works.md) | the classifier, concept links, search, context-loss detection |
| [HTTP API](docs/api.md) | every endpoint, the graph JSON, importing existing chats |
| [The viewer](docs/viewer.md) | a tour of the chat, graph, map, and search |
| [Running it](docs/running.md) | model options, pgvector, Docker and hosting |
| [Development](docs/development.md) | layout, tests, known limits, what's next |

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

The classifier is graded separately by `eval_immediate_previous.py`,
`eval_cross_links.py` and `eval_concept_links.py`; see
[Development](docs/development.md#verification).
