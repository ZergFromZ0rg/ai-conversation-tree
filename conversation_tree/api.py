import hmac
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chatService import (
    analyzeConversation,
    createConversationEdge,
    createConversationSession,
    createManualConceptLink,
    deleteConversationSession,
    getConversationSession,
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
from .conceptIndex import relinkAllConceptLinks
from .config import Settings
from .conversationErrors import InvalidInputError, NotFoundError, ProviderError
from .graphService import analyzeImmediateRelationship
from .analysisService import (
    buildConversationReport,
    buildConversationSummary,
    buildLlmSummary,
    reportAsMarkdown,
    searchConversations,
)
from .contextService import buildContextHealth
from .ingestService import ingestConversation
from .lifecycle import init, warmup
from .providers import Replier, listAvailableModels, listOllamaLibrary, pullOllamaModel


frontendDistDir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
router = APIRouter()


@dataclass(frozen=True)
class ServerOptions:
    """What only the HTTP layer needs; the logic layer never sees these."""

    apiToken: str = ""
    """Hosted deployments set a shared token; every API route then requires it
    (`Authorization: Bearer <token>`). The static viewer shell, /healthz and the
    index stay public so the page can load and ask for the token. Empty (the
    local default) means no auth."""

    corsOrigins: tuple[str, ...] = ()
    """Extra browser origins (a hosted frontend on another domain). The Vite dev
    server origins are always allowed."""

    @classmethod
    def fromEnv(cls, environ=None) -> "ServerOptions":
        environ = os.environ if environ is None else environ
        return cls(
            apiToken=environ.get("AI_CONVERSATION_TREE_API_TOKEN", ""),
            corsOrigins=tuple(
                origin.strip()
                for origin in environ.get("AI_CONVERSATION_TREE_CORS_ORIGINS", "").split(",")
                if origin.strip()
            ),
        )


_publicPaths = ("/", "/ui", "/healthz")


# The services own every rule and signal failure with a domain exception
# (conversationErrors.py); this layer only maps each kind to a status code.
async def handleNotFound(request: Request, error: NotFoundError):
    return JSONResponse({"detail": str(error)}, status_code=status.HTTP_404_NOT_FOUND)


async def handleInvalidInput(request: Request, error: InvalidInputError):
    return JSONResponse({"detail": str(error)}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


async def handleProviderError(request: Request, error: ProviderError):
    return JSONResponse({"detail": f"Model provider error: {error}"}, status_code=status.HTTP_502_BAD_GATEWAY)


class CreateConversationRequest(BaseModel):
    title: str | None = None
    model: str | None = None


class CreateTurnRequest(BaseModel):
    userText: str = Field(min_length=1)
    model: str | None = None
    apiKey: str | None = None


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


class CapturedMessage(BaseModel):
    role: str = Field(min_length=1, max_length=20)
    text: str = Field(max_length=200_000)


class IngestConversationRequest(BaseModel):
    source: str
    externalId: str = Field(min_length=1, max_length=200)
    title: str | None = Field(default=None, max_length=500)
    url: str | None = Field(default=None, max_length=2000)
    messages: list[CapturedMessage] = Field(max_length=4000)


class LlmSummaryRequest(BaseModel):
    model: str | None = None
    apiKey: str | None = None


class PullOllamaModelRequest(BaseModel):
    model: str = Field(min_length=1)


@router.get("/")
def root():
    return {
        "message": "AI Conversation Tree API",
        "endpoints": [
            "GET /ui",
            "GET /models",
            "GET /ollama/library",
            "POST /ollama/pull",
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
            "GET /conversations/{conversationId}/summary",
            "GET /conversations/{conversationId}/report",
            "GET /conversations/{conversationId}/context-health",
            "POST /conversations/{conversationId}/summary/llm",
            "POST /ingest/conversations",
            "GET /healthz",
            "GET /search",
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


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/ui")
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


@router.get("/models")
def getModels():
    return listAvailableModels()


@router.get("/ollama/library")
def getOllamaLibrary():
    return listOllamaLibrary()


@router.post("/ollama/pull")
def pullOllama(req: PullOllamaModelRequest):
    def events():
        try:
            for progress in pullOllamaModel(req.model):
                yield f"data: {json.dumps(progress)}\n\n"
        except ProviderError as error:
            yield f"data: {json.dumps({'status': 'error', 'message': str(error)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
def createConversation(req: CreateConversationRequest):
    return createConversationSession(req.title, req.model)


@router.get("/conversations")
def listConversations():
    return listConversationSessions()


@router.get("/conversations/{conversationId}")
def getConversation(conversationId: int):
    return getConversationSession(conversationId)


@router.patch("/conversations/{conversationId}")
def updateConversation(conversationId: int, req: UpdateConversationRequest):
    return setConversationSessionModel(conversationId, req.model)


@router.delete("/conversations/{conversationId}", status_code=status.HTTP_204_NO_CONTENT)
def deleteConversation(conversationId: int):
    deleteConversationSession(conversationId)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/conversations/{conversationId}/turns", status_code=status.HTTP_201_CREATED)
def createTurn(conversationId: int, req: CreateTurnRequest):
    return processChatMessage(conversationId, req.userText, req.model, req.apiKey)


@router.post("/conversations/{conversationId}/turns/stream")
def createTurnStream(conversationId: int, req: CreateTurnRequest):
    # Called (not iterated) here: a bad conversation or model raises now and
    # becomes a real 404/422, before the response commits to streaming.
    stream = streamChatMessage(conversationId, req.userText, req.model, req.apiKey)

    def events():
        # The response has already started (200, text/event-stream) by the
        # time a provider failure can happen, so a failure here goes out as an
        # {"type": "error"} event rather than an HTTP status — there is no HTTP
        # status left to change at this point.
        try:
            for event in stream:
                yield f"data: {json.dumps(event)}\n\n"
        except ProviderError as error:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Model provider error: {error}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/conversations/{conversationId}/turns")
def getTurns(conversationId: int):
    return listConversationTurns(conversationId)


@router.post("/conversations/{conversationId}/analyze")
def analyze(conversationId: int):
    return analyzeConversation(conversationId)


@router.get("/conversations/{conversationId}/graph")
def getGraph(conversationId: int):
    return listConversationGraph(conversationId)


@router.get("/conversations/{conversationId}/concept-links")
def getConversationConceptLinks(conversationId: int):
    return listConversationConceptLinks(conversationId)


@router.get("/conversations/{conversationId}/summary")
def getConversationSummary(conversationId: int):
    return buildConversationSummary(conversationId)


@router.post("/conversations/{conversationId}/summary/llm")
def getLlmSummary(conversationId: int, req: LlmSummaryRequest):
    return buildLlmSummary(conversationId, req.model, req.apiKey)


@router.get("/conversations/{conversationId}/context-health")
def getContextHealth(conversationId: int):
    return buildContextHealth(conversationId)


@router.post("/ingest/conversations", status_code=status.HTTP_200_OK)
def ingestConversationEndpoint(req: IngestConversationRequest):
    return ingestConversation(
        req.source,
        req.externalId,
        [message.model_dump() for message in req.messages],
        req.title,
        req.url,
    )


@router.get("/conversations/{conversationId}/report")
def getConversationReport(conversationId: int, format: str = "markdown"):
    if format not in ("markdown", "json"):
        raise HTTPException(status_code=422, detail="format must be 'markdown' or 'json'.")
    report = buildConversationReport(conversationId)
    title = report["conversation"]["title"] or f"conversation-{conversationId}"
    safeTitle = re.sub(r"[^a-zA-Z0-9_-]+", "-", title).strip("-") or f"conversation-{conversationId}"
    if format == "json":
        return Response(
            content=json.dumps(report, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{safeTitle}.json"'},
        )
    return PlainTextResponse(
        reportAsMarkdown(report),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{safeTitle}.md"'},
    )


@router.get("/search")
def search(query: str, limit: int = 20, conversationId: int | None = None):
    return searchConversations(query, limit, conversationId)


@router.get("/concepts/graph")
def getConceptGraph():
    return listConceptGraph()


@router.post("/concepts/relink")
def relinkConcepts():
    return {"linkCount": relinkAllConceptLinks()}


@router.post("/concept-links", status_code=status.HTTP_201_CREATED)
def createConceptLink(req: CreateConceptLinkRequest):
    return createManualConceptLink(req.aConceptKey, req.bConceptKey, req.kind)


@router.delete("/concept-links/{linkId}", status_code=status.HTTP_204_NO_CONTENT)
def deleteConceptLink(linkId: int):
    removeConceptLink(linkId)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/edges", status_code=status.HTTP_201_CREATED)
def createEdge(req: CreateEdgeRequest):
    return createConversationEdge(
        conversationId=req.conversationId,
        fromTurnId=req.fromTurnId,
        toTurnId=req.toTurnId,
        label=req.label,
        confidence=req.confidence,
    )


@router.patch("/edges/{edgeId}")
def updateEdge(edgeId: int, req: UpdateEdgeRequest):
    return patchConversationEdge(edgeId, label=req.label, confidence=req.confidence)


@router.delete("/edges/{edgeId}", status_code=status.HTTP_204_NO_CONTENT)
def deleteEdge(edgeId: int):
    removeConversationEdge(edgeId)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/debug/immediate")
def debugImmediate(req: ImmediateDebugRequest):
    return analyzeImmediateRelationship(
        req.previousUserText,
        req.previousAiText,
        req.userText,
    )


def createApp(
    settings: Settings | None = None,
    options: ServerOptions | None = None,
    replier: Replier | None = None,
) -> FastAPI:
    """Build the HTTP app. `settings` and `options` default to the environment,
    `replier` to the built-in providers.

    Nothing is configured until the app starts: the lifespan hook calls
    ``init(settings)`` (database) and ``warmup()`` (embedding models). Serve it
    with ``uvicorn --factory conversation_tree.api:createApp``. A test client
    that skips lifespan events must call ``conversation_tree.init`` itself.
    """
    options = options or ServerOptions.fromEnv()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = init(settings, replier=replier)
        logging.getLogger("uvicorn.error").info("conversation_tree database: %s", active.dbPath)
        warmup()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def requireApiToken(request: Request, call_next):
        path = request.url.path
        if (
            not options.apiToken
            or request.method == "OPTIONS"
            or path in _publicPaths
            or path.startswith("/assets/")
        ):
            return await call_next(request)
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(supplied.encode(), options.apiToken.encode()):
            return JSONResponse({"detail": "Missing or invalid API token."}, status_code=401)
        return await call_next(request)

    # Added after the token middleware, so it is the outermost layer: even a 401
    # carries CORS headers and the viewer can read it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", *options.corsOrigins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frontendAssetsDir = frontendDistDir / "assets"
    if frontendAssetsDir.is_dir():
        # The built index.html references assets with absolute /assets/... URLs.
        app.mount("/assets", StaticFiles(directory=frontendAssetsDir), name="assets")

    app.add_exception_handler(NotFoundError, handleNotFound)
    app.add_exception_handler(InvalidInputError, handleInvalidInput)
    app.add_exception_handler(ProviderError, handleProviderError)
    app.include_router(router)
    return app
