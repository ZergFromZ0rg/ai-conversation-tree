# The Viewer

A tour of the built-in chat viewer at `/ui`: the transcript, the graph drawer, the workspace map, search, and summaries.

[← Back to the README](../README.md)

## Five-Minute Demo

This walks through everything with no API key. It imports four chats from
three sources through the ingest API.

```bash
# 1. backend + viewer (one-time install: see running.md)
AI_CONVERSATION_TREE_STUB_LLM=1 AI_CONVERSATION_TREE_CONTEXT_TOKENS=400 \
  venv/bin/python -m uvicorn --factory conversation_tree.api:createApp

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

![Search across chats: results grouped by conversation with source badges, question/answer tags, turn positions, and highlighted matches.](images/search.png)

![Summary and context health: the structured summary, an "At risk" verdict, token pressure against the context window, and the flagged turns, each one click from the transcript.](images/context-health.png)

## Interface

The viewer at `/ui` is a normal chat: a conversation list on the left, the
transcript in the middle, a composer at the bottom.

![The chat viewer: conversation list with source badges on the left, an imported chat in the middle with its context-window horizon and a context-loss warning, model picker and composer at the bottom.](images/chat-view.png)

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

![The workspace map: concepts from four conversations clustered by colour, joined by cross-conversation concept links.](images/workspace-map.png)

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
