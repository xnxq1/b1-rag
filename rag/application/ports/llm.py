from typing import Protocol

from rag.domain.entities import AnswerDraft, ScoredChunk


class AnswerGeneratorPort(Protocol):
    async def generate(self, question: str, context: list[ScoredChunk]) -> AnswerDraft: ...
