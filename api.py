from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from chatService import createConversationSession, listConversationSessions, loadConversationSession, processChatMessage
from db import initDb
from graphService import analyzeImmediateRelationship, reclassifyTurns, resetConversation, turns

frontendDistDir = Path(__file__).parent / "frontend" / "dist"

app = FastAPI()
initDb()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if frontendDistDir.exists():
    app.mount("/ui/assets", StaticFiles(directory=frontendDistDir / "assets"), name="ui-assets")

@app.get("/")
def root():
    return {"message": "AI Conversation Tree API", "endpoints": ["/chat", "/graph", "/ui"]}

@app.get("/ui")
def ui():
    indexPath = frontendDistDir / "index.html"
    if not indexPath.exists():
        return JSONResponse(
            {
                "error": "React frontend not built.",
                "next": [
                    "cd frontend",
                    "npm install",
                    "npm run build",
                ],
            },
            status_code=503,
        )
    return FileResponse(indexPath, media_type="text/html")

class ChatRequest(BaseModel):
    conversationId: int
    userText: str


class CreateConversationRequest(BaseModel):
    title: str | None = None

class ImmediateDebugRequest(BaseModel):
    previousUserText: str
    previousAiText: str
    userText: str

@app.post("/chat")
def chat(req: ChatRequest):
    return processChatMessage(req.conversationId, req.userText)


@app.post("/conversations")
def createConversation(req: CreateConversationRequest):
    return createConversationSession(req.title)


@app.get("/conversations")
def listConversations():
    return listConversationSessions()


@app.get("/conversations/{conversationId}")
def getConversation(conversationId: int):
    return loadConversationSession(conversationId)

@app.post("/reset")
def reset():
    resetConversation()
    return {"nodes": 0, "edges": 0}

@app.post("/reclassify")
def reclassify():
    rebuiltTurns = reclassifyTurns()
    edgeCount = sum(len(turn.semanticParents) for turn in rebuiltTurns)
    return {"nodes": len(rebuiltTurns), "edges": edgeCount}

@app.post("/debug/immediate")
def debugImmediate(req: ImmediateDebugRequest):
    return analyzeImmediateRelationship(
        req.previousUserText,
        req.previousAiText,
        req.userText,
    )

@app.get("/graph")
def graph():
    return {
        "nodes": [
            {"id": t.id, "userText": t.userText, "aiText": t.aiText, "conceptIds": t.conceptIds}
            for t in turns
        ],
        "edges": [
            {"from": parentId, "to": t.id, "type": rel, "confidence": confidence}
            for t in turns
            for parentId, rel, confidence in t.semanticParents
        ],
    }
