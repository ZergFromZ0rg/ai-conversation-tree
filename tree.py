class Turn:
    def __init__(self, id, root, embedding, userText, aiText, timestamp, timelineParent, semanticParents, conceptIds):
        self.id = id
        self.root = root
        self.embedding = embedding
        self.userText = userText
        self.aiText = aiText
        self.timestamp = timestamp
        self.timelineParent = timelineParent
        self.semanticParents: list[tuple[int, str, float]] = semanticParents
        self.conceptIds: list[int] = conceptIds

    def __str__(self):
        return (f"Turn(id={self.id}, user='{self.userText}', ai='{self.aiText}', "
                f"timelineParent={self.timelineParent}, semanticParents={self.semanticParents}, "
                f"conceptIds={self.conceptIds})")
