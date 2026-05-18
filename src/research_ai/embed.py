from sentence_transformers import SentenceTransformer


class QueryEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return vector.tolist()