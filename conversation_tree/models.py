from dataclasses import dataclass, field


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
