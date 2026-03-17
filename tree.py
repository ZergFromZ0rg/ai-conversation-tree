class Message:
    def __init__(self, id, text, timestamp, user, timeline_parent, concept_id):
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.user = user
        self.timeline_parent = timeline_parent
        self.concept_id = concept_id

    def __repr__(self):
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"Message(id={self.id}, user={self.user!r}, text={preview!r})"


class Concept:
    def __init__(self, id, name, description):
        self.id = id
        self.name = name
        self.description = description
        
        