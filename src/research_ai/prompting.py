from research_ai.models import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("title", "Untitled")
        source_uri = chunk.metadata.get("source_uri", "Unknown source")

        parts.append(
f"""[Source {i}]
Title: {title}
Chunk ID: {chunk.chunk_id}
Source URI: {source_uri}
Relevance Score: {chunk.score:.4f}
{chunk.text}""")

    return "\n\n" + ("-" * 80 + "\n\n").join(parts)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    if not query.strip():
        raise ValueError("Query must not be empty")

    context = format_context(chunks)

    prompt = f"""You are a retrieval-augmented assistant.
Answer the user's question using only the provided context.
If the answer is not supported by the context, say that the available sources do not provide enough information.
Do not invent facts that are not grounded in the sources.
When possible, refer to the source numbers in your answer.

User Question:
{query}

Retrieved Context:
{context}

Answer:"""

    return prompt