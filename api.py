from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chatService import (
    analyzeConversation,
    createConversationEdge,
    createConversationSession,
    deleteConversationSession,
    getConversationSession,
    listConversationGraph,
    listConversationSessions,
    listConversationTurns,
    patchConversationEdge,
    processChatMessage,
    removeConversationEdge,
)
from db import getConversationTurnIds, initDb
from graphService import analyzeImmediateRelationship

EDGE_LABELS = ("continuation", "branch", "related")


app = FastAPI()
initDb()
frontendDistDir = Path(__file__).parent / "frontend" / "dist"

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
    app.mount("/assets", StaticFiles(directory=frontendDistDir / "assets"), name="assets")
    app.mount("/ui/assets", StaticFiles(directory=frontendDistDir / "assets"), name="ui-assets")


class CreateConversationRequest(BaseModel):
    title: str | None = None


class CreateTurnRequest(BaseModel):
    userText: str = Field(min_length=1)


class ImmediateDebugRequest(BaseModel):
    previousUserText: str
    previousAiText: str
    userText: str


class CreateEdgeRequest(BaseModel):
    conversationId: int
    fromTurnId: int
    toTurnId: int
    label: str
    confidence: float


class UpdateEdgeRequest(BaseModel):
    label: str | None = None
    confidence: float | None = None


@app.get("/")
def root():
    return {
        "message": "AI Conversation Tree API",
        "endpoints": [
            "GET /ui",
            "POST /conversations",
            "GET /conversations",
            "GET /conversations/{conversationId}",
            "DELETE /conversations/{conversationId}",
            "POST /conversations/{conversationId}/turns",
            "GET /conversations/{conversationId}/turns",
            "POST /conversations/{conversationId}/analyze",
            "GET /conversations/{conversationId}/graph",
            "POST /edges",
            "PATCH /edges/{edgeId}",
            "DELETE /edges/{edgeId}",
            "POST /debug/immediate",
        ],
    }


@app.get("/ui")
def ui():
    indexPath = frontendDistDir / "index.html"
    if not indexPath.exists():
        return JSONResponse(
            {
                "error": "Frontend not built.",
                "next": [
                    "cd frontend",
                    "npm install",
                    "npm run build",
                ],
            },
            status_code=503,
        )
    return FileResponse(indexPath, media_type="text/html")


@app.post("/conversations", status_code=status.HTTP_201_CREATED)
def createConversation(req: CreateConversationRequest):
    return createConversationSession(req.title)


@app.get("/conversations")
def listConversations():
    return listConversationSessions()


@app.get("/conversations/{conversationId}")
def getConversation(conversationId: int):
    conversation = getConversationSession(conversationId)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.delete("/conversations/{conversationId}", status_code=status.HTTP_204_NO_CONTENT)
def deleteConversation(conversationId: int):
    deleted = deleteConversationSession(conversationId)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/conversations/{conversationId}/turns", status_code=status.HTTP_201_CREATED)
def createTurn(conversationId: int, req: CreateTurnRequest):
    if getConversationSession(conversationId) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return processChatMessage(conversationId, req.userText)


@app.get("/conversations/{conversationId}/turns")
def getTurns(conversationId: int):
    if getConversationSession(conversationId) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return listConversationTurns(conversationId)


@app.post("/conversations/{conversationId}/analyze")
def analyze(conversationId: int):
    if getConversationSession(conversationId) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return analyzeConversation(conversationId)


@app.get("/conversations/{conversationId}/graph")
def getGraph(conversationId: int):
    if getConversationSession(conversationId) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return listConversationGraph(conversationId)


@app.post("/edges", status_code=status.HTTP_201_CREATED)
def createEdge(req: CreateEdgeRequest):
    if getConversationSession(req.conversationId) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if req.label not in EDGE_LABELS:
        raise HTTPException(status_code=422, detail=f"label must be one of {list(EDGE_LABELS)}.")
    if not 0.0 <= req.confidence <= 1.0:
        raise HTTPException(status_code=422, detail="confidence must be between 0.0 and 1.0.")
    if req.fromTurnId == req.toTurnId:
        raise HTTPException(status_code=422, detail="fromTurnId and toTurnId must differ.")
    if req.fromTurnId > req.toTurnId:
        raise HTTPException(
            status_code=422,
            detail="fromTurnId must be earlier than toTurnId; edges point from the earlier turn to the later turn.",
        )
    turnIds = set(getConversationTurnIds(req.conversationId))
    missing = [turnId for turnId in (req.fromTurnId, req.toTurnId) if turnId not in turnIds]
    if missing:
        raise HTTPException(status_code=404, detail=f"Turns not found in conversation: {missing}.")
    return createConversationEdge(
        conversationId=req.conversationId,
        fromTurnId=req.fromTurnId,
        toTurnId=req.toTurnId,
        label=req.label,
        confidence=req.confidence,
    )


@app.patch("/edges/{edgeId}")
def updateEdge(edgeId: int, req: UpdateEdgeRequest):
    if req.label is not None and req.label not in EDGE_LABELS:
        raise HTTPException(status_code=422, detail=f"label must be one of {list(EDGE_LABELS)}.")
    if req.confidence is not None and not 0.0 <= req.confidence <= 1.0:
        raise HTTPException(status_code=422, detail="confidence must be between 0.0 and 1.0.")
    edge = patchConversationEdge(edgeId, label=req.label, confidence=req.confidence)
    if edge is None:
        raise HTTPException(status_code=404, detail="Edge not found.")
    return edge


@app.delete("/edges/{edgeId}", status_code=status.HTTP_204_NO_CONTENT)
def deleteEdge(edgeId: int):
    deleted = removeConversationEdge(edgeId)
    if not deleted:
        raise HTTPException(status_code=404, detail="Edge not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/debug/immediate")
def debugImmediate(req: ImmediateDebugRequest):
    return analyzeImmediateRelationship(
        req.previousUserText,
        req.previousAiText,
        req.userText,
    )
