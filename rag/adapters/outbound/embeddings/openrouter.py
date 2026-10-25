from langchain_openai import OpenAIEmbeddings

from rag.adapters.outbound.openai_errors import translate_openai_errors


class OpenRouterEmbeddings:
    def __init__(self, embeddings: OpenAIEmbeddings, dimension: int) -> None:
        self._embeddings = embeddings
        self.dimension = dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with translate_openai_errors("Сервис эмбеддингов"):
            vectors = await self._embeddings.aembed_documents(texts)
        for vector in vectors:
            self._check_dimension(vector)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        with translate_openai_errors("Сервис эмбеддингов"):
            vector = await self._embeddings.aembed_query(text)
        self._check_dimension(vector)
        return vector

    def _check_dimension(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            msg = f"Модель эмбеддингов вернула вектор размерности {len(vector)}, а EMBEDDING_DIM={self.dimension}"
            raise RuntimeError(msg)
