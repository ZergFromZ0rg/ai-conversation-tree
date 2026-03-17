class Message:
    def __init__(self, id, root, embedding, text, timestamp, user, timeline_parent, semantic_parent, concept_id):
        self.id = id
        self.root = root
        self.embedding = embedding
        self.text = text
        self.timestamp = timestamp
        self.user = user
        self.timeline_parent = timeline_parent
        self.semantic_parents: list[tuple[int, str]] = semantic_parent
        self.concept_ids: list[int] = concept_id
    
    def __str__(self):
        return f"Message(id={self.id}, user={self.user}, text='{self.text}'), timeline_parent={self.timeline_parent}, semantic_parents={self.semantic_parents}, concept_ids={self.concept_ids})"



class Concept:
    def __init__(self, id, name, description):
        self.id = id
        self.name = name
        self.description = description
        
        