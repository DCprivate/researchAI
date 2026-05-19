from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkRecord:
    artifact_version: str
    chunk_id: str
    doc_id: str
    text: str
    token_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    artifact_version: str
    chunk_id: str
    doc_id: str
    #text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    artifact_version: str
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)