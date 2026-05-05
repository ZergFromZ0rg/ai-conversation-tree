from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from main import addTurn, analyzeImmediateRelationship, reclassifyTurns, resetConversation, turns

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Conversation Tree API", "endpoints": ["/chat", "/graph", "/ui"]}

@app.get("/ui")
def ui():
    return FileResponse("index.html", media_type="text/html")

class ChatRequest(BaseModel):
    userText: str
    aiText: str

class ImmediateDebugRequest(BaseModel):
    previousUserText: str
    previousAiText: str
    userText: str

@app.post("/chat")
def chat(req: ChatRequest):
    turn = addTurn(req.userText, req.aiText)
    return {"turnId": turn.id, "conceptIds": turn.conceptIds, "semanticParents": turn.semanticParents}

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
    nodes = [
        {"id": t.id, "userText": t.userText, "aiText": t.aiText, "conceptIds": t.conceptIds}
        for t in turns
    ]
    edges = [
        {"from": parentId, "to": t.id, "type": rel, "confidence": confidence}
        for t in turns
        for parentId, rel, confidence in t.semanticParents
    ]
    return {"nodes": nodes, "edges": edges}
