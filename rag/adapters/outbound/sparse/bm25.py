import asyncio
import re

from fastembed import SparseTextEmbedding

from rag.application.ports.sparse import SparseVector

_COMPOUND_NUMBER = re.compile(r"\d+(?:[.,/\-]\d+)+")
_GROUPED_NUMBER = re.compile(r"(?<!\d)\d{1,3}(?:[  ]\d{3})+(?!\d)")
_SEPARATORS = re.compile(r"[.,/\-]")


def add_exact_tokens(text: str) -> str:
    extra = [_SEPARATORS.sub("_", m.group()) for m in _COMPOUND_NUMBER.finditer(text)]
    extra += [re.sub(r"\s", "", m.group()) for m in _GROUPED_NUMBER.finditer(text)]
    if not extra:
        return text
    return f"{text}\n{' '.join(extra)}"


class FastEmbedBm25Encoder:
    def __init__(self, language: str, avg_len: float) -> None:
        self._model = SparseTextEmbedding("Qdrant/bm25", language=language, avg_len=avg_len)

    async def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return await asyncio.to_thread(self._encode_documents, texts)

    async def encode_query(self, text: str) -> SparseVector:
        return await asyncio.to_thread(self._encode_query, text)

    def _encode_documents(self, texts: list[str]) -> list[SparseVector]:
        prepared = [add_exact_tokens(t) for t in texts]
        return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in self._model.embed(prepared)]

    def _encode_query(self, text: str) -> SparseVector:
        embedding = next(iter(self._model.query_embed(add_exact_tokens(text))))
        return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
