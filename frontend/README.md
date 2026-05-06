# Frontend

This folder is the Phase 1 React UI.

## Purpose

- keep the current `FastAPI` backend
- replace the single `index.html` UI with a React app
- preserve the chat-first layout:
  - center chat thread
  - left conversation drawer
  - right graph drawer

## Expected Commands

```bash
npm install
npm run dev
```

The Vite dev server runs on `http://127.0.0.1:5173` and proxies API requests to the backend on `http://127.0.0.1:8000`.

## Structure

- `src/app.tsx`
  - top-level app shell and state management
- `src/components/conversationSidebar.tsx`
  - saved chat history drawer
- `src/components/chatThread.tsx`
  - linear chat view
- `src/components/chatComposer.tsx`
  - bottom composer
- `src/components/graphDrawer.tsx`
  - graph panel using `vis-network`
- `src/lib/api.ts`
  - typed API client
- `src/types.ts`
  - shared frontend types
