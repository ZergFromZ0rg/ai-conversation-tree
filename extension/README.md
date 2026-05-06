# Extension Plan

This folder is a placeholder for the browser integration path.

## Goal

Let a user keep chatting on an existing AI site and attach the graph UI beside it, without paying for extra response-generation API calls through this project.

## Recommended Future Structure

- `extension/manifest.json`
  - browser extension manifest
- `extension/src/content/`
  - content scripts that run on supported AI chat sites
- `extension/src/adapters/`
  - site-specific parsers for:
    - ChatGPT
    - Claude
    - Gemini
- `extension/src/panel/`
  - injected React side panel UI
- `extension/src/storage/`
  - browser-local persistence, likely `IndexedDB`

## Backend Relationship

Recommended path:

1. browser extension extracts visible chat turns from the page
2. extension sends those turns to the local `FastAPI` backend on `localhost`
3. backend runs embeddings, classification, and graph construction
4. extension renders the graph as a right-side drawer

This keeps the existing Python graph logic intact while avoiding additional LLM response-generation costs.
