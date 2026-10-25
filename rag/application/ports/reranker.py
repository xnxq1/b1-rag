from typing import Protocol

from rag.domain.entities import ScoredChunk


class RerankerPort(Protocol):
    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...
