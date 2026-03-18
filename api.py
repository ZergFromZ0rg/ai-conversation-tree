from fastapi import FastAPI
from pydantic import BaseModel
from main import add_turn, turns

app = FastAPI()

class ChatRequest(BaseModel):
    user_text: str
    ai_text: str

@app.post("/chat")
def chat(req: ChatRequest):
    turn = add_turn(req.user_text, req.ai_text)
    return {"turn_id": turn.id, "concept_ids": turn.concept_ids, "semantic_parents": turn.semantic_parents}

@app.get("/graph")
def graph():
    nodes = [
        {"id": t.id, "user_text": t.user_text, "ai_text": t.ai_text, "concept_ids": t.concept_ids}
        for t in turns
    ]
    edges = [
        {"from": parent_id, "to": t.id, "type": rel}
        for t in turns
        for parent_id, rel in t.semantic_parents
    ]
    return {"nodes": nodes, "edges": edges}