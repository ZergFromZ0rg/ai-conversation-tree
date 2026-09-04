import json
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chatService import (
    analyzeConversation,
    createConversationEdge,
    createConversationSession,
    createManualConceptLink,
    deleteConversationSession,
    getConversationSession,
    isValidModelSpec,
    listAvailableModels,
    listConceptGraph,
    listConversationConceptLinks,
    listConversationGraph,
    listConversationSessions,
    listConversationTurns,
    patchConversationEdge,
    processChatMessage,
    removeConceptLink,
    removeConversationEdge,
    setConversationSessionModel,
    streamChatMessage,
)
from conceptIndex import relinkAllConceptLinks
from db import getConversationTurnIds, initDb
from graphService import analyzeImmediateRelationship

EDGE_LABELS = ("continuation", "branch", "related")
CONCEPT_LINK_KINDS = ("same", "related")


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

frontendAssetsDir = frontendDistDir / "assets"
if frontendAssetsDir.is_dir():
    # The built index.html references assets with absolute /assets/... URLs.
    app.mount("/assets", StaticFiles(directory=frontendAssetsDir), name="assets")


class CreateConversationRequest(BaseModel):
    title: str | None = None
    model: str | None = None


class CreateTurnRequest(BaseModel):
    userText: str = Field(min_length=1)
    model: str | None = None


class UpdateConversationRequest(BaseModel):
    model: str | None = None


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


class CreateConceptLinkRequest(BaseModel):
    aConceptKey: str = Field(min_length=1)
    bConceptKey: str = Field(min_length=1)
    kind: str = "related"


@app.get("/")
def root():
    return {
        "message": "AI Conversation Tree API",
        "endpoints": [
            "GET /ui",
            "GET /models",
            "POST /conversations",
            "GET /conversations",
            "GET /conversations/{conversationId}",
            "PATCH /conversations/{conversationId}",
            "DELETE /conversations/{conversationId}",
            "POST /conversations/{conversationId}/turns",
            "POST /conversations/{conversationId}/turns/stream",
            "GET /conversations/{conversationId}/turns",
            "POST /conversations/{conversationId}/analyze",
            "GET /conversations/{conversationId}/graph",
            "GET /conversations/{conversationId}/concept-links",
            "GET /concepts/graph",
            "POST /concepts/relink",
            "POST /concept-links",
            "DELETE /concept-links/{linkId}",
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


@app.get("/models")
def getModels():
    return listAvailableModels()


@app.post("/conversations", status_code=status.HTTP_201_CREATED)
def createConversation(req: CreateConversationRequest):
    if req.model is not None and not isValidModelSpec(req.model):
        raise HTTPException(
            status_code=422,
            detail="model must be 'stub', 'ollama:<name>', or 'openai:<name>'.",
        )
    return createConversationSession(req.title, req.model)


@app.get("/conversations")
def listConversations():
    return listConversationSessions()


@app.get("/conversations/{conversationId}")
def getConversation(conversationId: int):
    conversation = getConversationSession(conversationId)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conversation


@app.patch("/conversations/{conversationId}")
def updateConversation(conversationId: int, req: UpdateConversationRequest):
    if req.model is not None and not isValidModelSpec(req.model):
        raise HTTPException(
            status_code=422,
            detail="model must be 'stub', 'ollama:<name>', or 'openai:<name>'.",
        )
    conversation = setConversationSessionModel(conversationId, req.model)
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
    if req.model is not None and not isValidModelSpec(req.model):
        raise HTTPException(
            status_code=422,
            detail="model must be 'stub', 'ollama:<name>', or 'openai:<name>'.",
        )
    try:
        return processChatMessage(conversationId, req.userText, req.model)
    except (httpx.HTTPError, RuntimeError) as error:
        raise HTTPException(status_code=502, detail=f"Model provider error: {error}") from error


@app.post("/conversations/{conversationId}/turns/stream")
def createTurnStream(conversationId: int, req: CreateTurnRequest):
    if getConversationSession(conversationId) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if req.model is not None and not isValidModelSpec(req.model):
        raise HTTPException(
            status_code=422,
            detail="model must be 'stub', 'ollama:<name>', or 'openai:<name>'.",
        )

    def events():
        # The response has already started (200, text/event-stream) by the
        # time a provider or persistence failure can happen, so a failure here
        # goes out as an {"type": "error"} event rather than an HTTP status —
        # there is no HTTP status left to change at this point.
        try:
            for event in streamChatMessage(conversationId, req.userText, req.model):
                yield f"data: {json.dumps(event)}\n\n"
        except (httpx.HTTPError, RuntimeError) as error:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Model provider error: {error}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


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


@app.get("/conversations/{conversationId}/concept-links")
def getConversationConceptLinks(conversationId: int):
    if getConversationSession(conversationId) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return listConversationConceptLinks(conversationId)


@app.get("/concepts/graph")
def getConceptGraph():
    return listConceptGraph()


@app.post("/concepts/relink")
def relinkConcepts():
    return {"linkCount": relinkAllConceptLinks()}


@app.post("/concept-links", status_code=status.HTTP_201_CREATED)
def createConceptLink(req: CreateConceptLinkRequest):
    if req.kind not in CONCEPT_LINK_KINDS:
        raise HTTPException(status_code=422, detail=f"kind must be one of {list(CONCEPT_LINK_KINDS)}.")
    if req.aConceptKey == req.bConceptKey:
        raise HTTPException(status_code=422, detail="aConceptKey and bConceptKey must differ.")
    link = createManualConceptLink(req.aConceptKey, req.bConceptKey, req.kind)
    if link is None:
        raise HTTPException(status_code=404, detail="One or both concepts were not found.")
    return link


@app.delete("/concept-links/{linkId}", status_code=status.HTTP_204_NO_CONTENT)
def deleteConceptLink(linkId: int):
    deleted = removeConceptLink(linkId)
    if not deleted:
        raise HTTPException(status_code=404, detail="Concept link not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
