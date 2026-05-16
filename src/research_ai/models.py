from dataclasses import dataclass, field
from typing import Any

@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    token_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    chunk_id: str
    doc_id: str
    #text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)