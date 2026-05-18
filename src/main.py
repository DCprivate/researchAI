import json
from pathlib import Path

from research_ai.io import load_chunks_jsonl, load_embeddings_jsonl, build_chunk_map
from research_ai.retriever import retrieve_top_k
from research_ai.embed import QueryEmbedder
from research_ai.prompting import build_prompt
from research_ai.llm import OllamaClient


def main():
    
    path = Path(r"C:\Users\devyn\Desktop\g\dataset-builder\output")
    
    chunks = load_chunks_jsonl(path / "chunks.jsonl")
    embeddings = load_embeddings_jsonl(path / "embeddings.jsonl")
    chunk_map = build_chunk_map(chunks)

    """print(f"loaded chunks: {len(chunks)}")
    print(f"loaded embeddings: {len(embeddings)}")
    print(f"chunk map size: {len(chunk_map)}")

    first_embedding = embeddings[1]
    linked_chunk = chunk_map[first_embedding.chunk_id]

    print("=" * 80)
    print(f"embedding chunk_id: {first_embedding.chunk_id}")
    print(f"linked text preview: {linked_chunk.text[:200]}")
    print(f"linked metadata: {str(linked_chunk.metadata)[:200]}")
    
    chunk_text = chunk_map[some_embedding.chunk_id].text"""
    
    # temporary test: use one existing embedding as a fake query
    #query_embedding = embeddings[0].embedding
    
    embedder = QueryEmbedder()
    query = "is there anything in the documents about pollutants? don't give anything about c++ or from the primer book"
    query_embedding = embedder.embed_query(query)

    results = retrieve_top_k(query_embedding=query_embedding, embedded_chunks=embeddings, chunk_map=chunk_map, k=500)

    """print("Query:", query)
    for result in results:
        print("=" * 80)
        print(f"score:    {result.score:.4f}")
        print(f"chunk_id: {result.chunk_id}")
        print(f"title:    {result.metadata.get('title')}")
        print(f"text:     {result.text[:300]}")"""
        
    for result in results:
        print(result.metadata)

    prompt = build_prompt(query, results)
    #print(prompt)
    
    llm = OllamaClient(model_name="mistral:latest")
    answer = llm.generate(prompt)
    
    print("\nAnswer:\n")
    print(answer)

if __name__ == "__main__":
    main()