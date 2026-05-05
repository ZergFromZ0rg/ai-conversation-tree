from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from main import add_turn, analyze_immediate_relationship, reclassify_turns, reset_conversation, turns

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Conversation Tree API", "endpoints": ["/chat", "/graph", "/ui"]}

@app.get("/ui")
def ui():
    return FileResponse("index.html", media_type="text/html")

class ChatRequest(BaseModel):
    user_text: str
    ai_text: str

class ImmediateDebugRequest(BaseModel):
    previous_user_text: str
    previous_ai_text: str
    user_text: str

@app.post("/chat")
def chat(req: ChatRequest):
    turn = add_turn(req.user_text, req.ai_text)
    return {"turn_id": turn.id, "concept_ids": turn.concept_ids, "semantic_parents": turn.semantic_parents}

@app.post("/reset")
def reset():
    reset_conversation()
    return {"nodes": 0, "edges": 0}

@app.post("/reclassify")
def reclassify():
    rebuilt_turns = reclassify_turns()
    edge_count = sum(len(turn.semantic_parents) for turn in rebuilt_turns)
    return {"nodes": len(rebuilt_turns), "edges": edge_count}

@app.post("/debug/immediate")
def debug_immediate(req: ImmediateDebugRequest):
    return analyze_immediate_relationship(
        req.previous_user_text,
        req.previous_ai_text,
        req.user_text,
    )

@app.get("/graph")
def graph():
    nodes = [
        {"id": t.id, "user_text": t.user_text, "ai_text": t.ai_text, "concept_ids": t.concept_ids}
        for t in turns
    ]
    edges = [
        {"from": parent_id, "to": t.id, "type": rel, "confidence": confidence}
        for t in turns
        for parent_id, rel, confidence in t.semantic_parents
    ]
    return {"nodes": nodes, "edges": edges}
