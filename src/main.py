import json
from pathlib import Path

from research_ai.io import load_chunks_jsonl, load_embeddings_jsonl, build_chunk_map


def main():
    
    path = Path(r"C:\Users\devyn\Desktop\g\dataset-builder\output")
    
    chunks = load_chunks_jsonl(path / "chunks.jsonl")
    embeddings = load_embeddings_jsonl(path / "embeddings.jsonl")
    chunk_map = build_chunk_map(chunks)

    print(f"loaded chunks: {len(chunks)}")
    print(f"loaded embeddings: {len(embeddings)}")
    print(f"chunk map size: {len(chunk_map)}")

    first_embedding = embeddings[0]
    linked_chunk = chunk_map[first_embedding.chunk_id]

    print("=" * 80)
    print(f"embedding chunk_id: {first_embedding.chunk_id}")
    print(f"linked text preview: {linked_chunk.text[:200]}")
    print(f"linked metadata: {str(linked_chunk.metadata)[:200]}")
    
    #chunk_text = chunk_map[some_embedding.chunk_id].text


if __name__ == "__main__":
    main()