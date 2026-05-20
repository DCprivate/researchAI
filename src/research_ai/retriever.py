import math

from research_ai.models import ChunkRecord, ChunkEmbedding, RetrievedChunk
from research_ai.schema import SUPPORTED_ARTIFACT_VERSIONS

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector dimensions do not match")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def retrieve_top_k(
    query_embedding: list[float],
    embedded_chunks: list[ChunkEmbedding],
    chunk_map: dict[str, ChunkRecord], k: int = 5,
    ) -> list[RetrievedChunk]:
    
    scored: list[tuple[float, ChunkEmbedding]] = []
    
    for embedded_chunk in embedded_chunks:
        score = cosine_similarity(query_embedding, embedded_chunk.embedding)
        scored.append((score, embedded_chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = scored[:k]

    results: list[RetrievedChunk] = []

    for score, embedded_chunk in top_k:
        chunk = chunk_map.get(embedded_chunk.chunk_id)
        if chunk is None:
            raise ValueError(
                f"Missing chunk text for chunk_id: {embedded_chunk.chunk_id}"
            )

        results.append(
            RetrievedChunk(
                artifact_version=SUPPORTED_ARTIFACT_VERSIONS,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=score,
                metadata=chunk.metadata,
            )
        )

    return results