import json
from pathlib import Path

from research_ai.io import load_chunks_jsonl, load_embeddings_jsonl, build_chunk_map
from research_ai.retriever import retrieve_top_k
from research_ai.prompting import build_prompt
from research_ai.embed import QueryEmbedder
from research_ai.llm import OllamaClient


def main():
    
    path = Path(r"C:\Users\devyn\Desktop\g\dataset-builder\output")
    
    # load chunks/embeddings and map them based on ID
    try:
        chunks = load_chunks_jsonl(path / "chunks.jsonl")
        embeddings = load_embeddings_jsonl(path / "embeddings.jsonl")
        chunk_map = build_chunk_map(chunks)
    except Exception as exc:
        print(f"[ERROR] Artifact validation failed: {exc}")
        return
    
    # Set up query (this is hardcoded for now)
    embedder = QueryEmbedder()
    query = "I want you to explain to me how blackholes work"
    query_embedding = embedder.embed_query(query)

    results = retrieve_top_k(query_embedding=query_embedding, embedded_chunks=embeddings, chunk_map=chunk_map, k=5)
    
    """for result in results:
        print("="*80)
        print(result.score)
        print(result.metadata)
        print(result.text)"""

    prompt = build_prompt(query, results)

    llm = OllamaClient(model_name="mistral:latest")
    answer = llm.generate(prompt)
    
    print("\nAnswer:\n")
    print(answer)

if __name__ == "__main__":
    main()