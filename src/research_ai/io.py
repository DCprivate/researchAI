import json
from pathlib import Path

from research_ai.models import ChunkRecord, EmbeddedChunk


SUPPORTED_ARTIFACT_VERSIONS = {"1.0"}


def read_jsonl(path: str | Path) -> list[dict]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")

    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc

    return records


def validate_chunk_row(row: dict) -> None:
    required = {
        #"artifact_version": str,
        "chunk_id": str,
        "doc_id": str,
        "text": str,
        "token_estimate": int,
        "metadata": dict,
    }

    for key, expected_type in required.items():
        if key not in row:
            raise ValueError(f"Chunk row missing required field: {key}")

        if not isinstance(row[key], expected_type):
            raise TypeError(
                f"Chunk field '{key}' must be {expected_type.__name__}, got {type(row[key]).__name__}"
            )

    """if row["artifact_version"] not in SUPPORTED_ARTIFACT_VERSIONS:
        raise ValueError(
            f"Unsupported chunk artifact_version: {row['artifact_version']}"
        )"""


def validate_embedding_row(row: dict) -> None:
    required = {
        #"artifact_version": str,
        "chunk_id": str,
        "doc_id": str,
        "embedding": list,
        "metadata": dict,
    }

    for key, expected_type in required.items():
        if key not in row:
            raise ValueError(f"Embedding row missing required field: {key}")

        if not isinstance(row[key], expected_type):
            raise TypeError(
                f"Embedding field '{key}' must be {expected_type.__name__}, got {type(row[key]).__name__}"
            )

    """if row["artifact_version"] not in SUPPORTED_ARTIFACT_VERSIONS:
        raise ValueError(
            f"Unsupported embedding artifact_version: {row['artifact_version']}"
        )"""

    if not row["embedding"]:
        raise ValueError("Embedding vector must not be empty")

    for value in row["embedding"]:
        if not isinstance(value, (int, float)):
            raise TypeError("Embedding vector elements must be numeric")


def load_chunks_jsonl(path: str | Path) -> list[ChunkRecord]:
    rows = read_jsonl(path)

    chunks = []
    for row in rows:
        validate_chunk_row(row)
        chunks.append(ChunkRecord(**row))

    return chunks


def load_embeddings_jsonl(path: str | Path) -> list[EmbeddedChunk]:
    rows = read_jsonl(path)

    embeddings = []
    expected_dim = None

    for row in rows:
        validate_embedding_row(row)

        current_dim = len(row["embedding"])
        if expected_dim is None:
            expected_dim = current_dim
        elif current_dim != expected_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions: expected {expected_dim}, got {current_dim}"
            )

        embeddings.append(EmbeddedChunk(**row))

    return embeddings


def build_chunk_map(chunks: list[ChunkRecord]) -> dict[str, ChunkRecord]:
    chunk_map: dict[str, ChunkRecord] = {}

    for chunk in chunks:
        if chunk.chunk_id in chunk_map:
            raise ValueError(f"Duplicate chunk_id found: {chunk.chunk_id}")
        chunk_map[chunk.chunk_id] = chunk

    return chunk_map