from dataclasses import dataclass
from typing import Protocol

from rag.application.ports.sparse import SparseVector
from rag.domain.entities import Chunk, Document, ScoredChunk


@dataclass(slots=True)
class HybridSearchResult:
    candidates: list[ScoredChunk]
    dense_hits: int | None = None
    sparse_hits: int | None = None


class VectorStorePort(Protocol):
    async def ensure_collection(self) -> None: ...

    async def upsert_document(
        self,
        document: Document,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVector],
    ) -> None: ...

    async def delete_document(self, doc_id: str) -> None: ...

    async def document_exists(self, doc_id: str) -> bool: ...

    async def list_documents(self) -> list[Document]: ...

    async def hybrid_search(
        self,
        doc_id: str,
        dense: list[float],
        sparse: SparseVector,
        limit: int,
        with_diagnostics: bool = False,
    ) -> HybridSearchResult: ...
