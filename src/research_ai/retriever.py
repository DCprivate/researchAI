import math

from research_ai.models import ChunkRecord, EmbeddedChunk, RetrievedChunk

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """_summary_

    Args:
        vec_a (list[float]): _description_
        vec_b (list[float]): _description_

    Raises:
        ValueError: _description_

    Returns:
        float: _description_
    """
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
    embedded_chunks: list[EmbeddedChunk],
    chunk_map: dict[str, ChunkRecord], k: int = 5,
    ) -> list[RetrievedChunk]:
    """_summary_

    Args:
        query_embedding (list[float]): _description_
        embedded_chunks (list[EmbeddedChunk]): _description_
        chunk_map (dict[str, ChunkRecord]): _description_
        k (int, optional): _description_. Defaults to 5.

    Raises:
        ValueError: _description_

    Returns:
        list[RetrievedChunk]: _description_
    """
    
    scored: list[tuple[float, EmbeddedChunk]] = []
    
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
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=score,
                metadata=chunk.metadata,
            )
        )

    return results