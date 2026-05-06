from dataclasses import dataclass, field


@dataclass
class ConversationModel:
    id: int
    title: str | None
    createdAt: str
    updatedAt: str


@dataclass
class EdgeModel:
    fromTurnId: int
    toTurnId: int
    label: str
    confidence: float


@dataclass
class TurnModel:
    id: int
    root: bool
    embedding: object
    userText: str
    aiText: str
    timestamp: object
    timelineParent: int | None
    semanticParents: list[tuple[int, str, float]] = field(default_factory=list)
    conceptIds: list[int] = field(default_factory=list)
