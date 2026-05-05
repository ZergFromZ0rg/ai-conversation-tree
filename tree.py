class Turn:
    def __init__(self, id, root, embedding, user_text, ai_text, timestamp, timeline_parent, semantic_parents, concept_ids):
        self.id = id
        self.root = root
        self.embedding = embedding
        self.user_text = user_text
        self.ai_text = ai_text
        self.timestamp = timestamp
        self.timeline_parent = timeline_parent
        self.semantic_parents: list[tuple[int, str, float]] = semantic_parents
        self.concept_ids: list[int] = concept_ids

    def __str__(self):
        return (f"Turn(id={self.id}, user='{self.user_text}', ai='{self.ai_text}', "
                f"timeline_parent={self.timeline_parent}, semantic_parents={self.semantic_parents}, "
                f"concept_ids={self.concept_ids})")
