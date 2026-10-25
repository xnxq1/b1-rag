import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from functools import partial
from http import HTTPStatus
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import httpx
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.adapters.outbound.retry import RetryPolicy
from rag.application.ports.sparse import SparseVector
from rag.application.ports.vector_store import HybridSearchResult
from rag.domain.entities import Chunk, Document, Heading, PageSpan, ScoredChunk

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

_SERVICE = "Векторное хранилище"
_LIST_PAGE_SIZE = 256
_DOCUMENT_FIELDS = ["doc_id", "doc_title", "filename", "chunks_total", "pages_total", "indexed_at"]


def point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"rag:{doc_id}:{chunk_index}"))


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ResponseHandlingException | httpx.TransportError):
        return True
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and (
            exc.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR or exc.status_code == HTTPStatus.TOO_MANY_REQUESTS
        )
    return False


_READY = models.FieldCondition(key="ready", match=models.MatchValue(value=True))


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])


def _ready_doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)), _READY])


class QdrantVectorStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        retry: RetryPolicy,
        collection: str,
        dense_dim: int,
        fusion: str,
        dense_limit: int,
        sparse_limit: int,
        upsert_batch_size: int,
    ) -> None:
        self._client = client
        self._retry = retry
        self._collection = collection
        self._dense_dim = dense_dim
        self._fusion = models.Fusion(fusion)
        self._dense_limit = dense_limit
        self._sparse_limit = sparse_limit
        self._batch_size = upsert_batch_size

    async def ensure_collection(self) -> None:
        await self._call(self._ensure_collection)

    async def _ensure_collection(self) -> None:
        if not await self._client.collection_exists(self._collection):
            try:
                await self._client.create_collection(
                    self._collection,
                    vectors_config={
                        DENSE_VECTOR: models.VectorParams(size=self._dense_dim, distance=models.Distance.COSINE),
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF),
                    },
                )
                logger.info("Создана коллекция %s", self._collection)
            except UnexpectedResponse as exc:
                if exc.status_code != HTTPStatus.CONFLICT:
                    raise

        info = await self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
        if dense is None or dense.size != self._dense_dim:
            actual = dense.size if dense else None
            msg = (
                f"Коллекция {self._collection}: размерность dense-вектора {actual}, а EMBEDDING_DIM={self._dense_dim}. "
                "При смене модели эмбеддингов нужна новая коллекция и переиндексация."
            )
            raise RuntimeError(msg)

        indexes: dict[str, Any] = {
            "doc_id": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
            "chunk_index": models.PayloadSchemaType.INTEGER,
            "section": models.PayloadSchemaType.KEYWORD,
            "page_from": models.PayloadSchemaType.INTEGER,
            "ready": models.PayloadSchemaType.BOOL,
        }
        existing = info.payload_schema
        for field, schema in indexes.items():
            if field not in existing:
                await self._client.create_payload_index(self._collection, field, field_schema=schema)

    async def upsert_document(
        self,
        document: Document,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[SparseVector],
    ) -> None:
        if not len(chunks) == len(dense) == len(sparse):
            msg = "Число фрагментов и векторов не совпадает"
            raise ValueError(msg)
        await self.delete_document(document.doc_id)

        points = [
            models.PointStruct(
                id=point_id(chunk.doc_id, chunk.chunk_index),
                vector={
                    DENSE_VECTOR: dense_vector,
                    SPARSE_VECTOR: models.SparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                },
                payload=self._payload(document, chunk),
            )
            for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse, strict=True)
        ]
        for start in range(0, len(points), self._batch_size):
            batch = points[start : start + self._batch_size]
            await self._call(partial(self._client.upsert, self._collection, points=batch, wait=True))
        await self._publish(document.doc_id)

    async def _publish(self, doc_id: str) -> None:
        await self._call(
            partial(self._client.set_payload, self._collection, payload={"ready": True}, points=_doc_filter(doc_id)),
        )

    async def delete_document(self, doc_id: str) -> None:
        selector = models.FilterSelector(filter=_doc_filter(doc_id))
        await self._call(lambda: self._client.delete(self._collection, points_selector=selector, wait=True))

    async def document_exists(self, doc_id: str) -> bool:
        result = await self._call(lambda: self._client.count(self._collection, count_filter=_ready_doc_filter(doc_id)))
        return result.count > 0

    async def list_documents(self) -> list[Document]:
        first_chunks = models.Filter(
            must=[models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0)), _READY],
        )
        documents: list[Document] = []
        offset = None
        while True:
            records, offset = await self._call(
                partial(
                    self._client.scroll,
                    self._collection,
                    scroll_filter=first_chunks,
                    limit=_LIST_PAGE_SIZE,
                    offset=offset,
                    with_payload=_DOCUMENT_FIELDS,
                ),
            )
            documents.extend(self._to_document(r.payload or {}) for r in records)
            if offset is None:
                return documents

    async def hybrid_search(
        self,
        doc_id: str,
        dense: list[float],
        sparse: SparseVector,
        limit: int,
        with_diagnostics: bool = False,
    ) -> HybridSearchResult:
        doc_filter = _ready_doc_filter(doc_id)
        prefetch = [models.Prefetch(query=dense, using=DENSE_VECTOR, limit=self._dense_limit, filter=doc_filter)]
        sparse_query = models.SparseVector(indices=sparse.indices, values=sparse.values)
        if sparse.indices:
            prefetch.append(
                models.Prefetch(query=sparse_query, using=SPARSE_VECTOR, limit=self._sparse_limit, filter=doc_filter),
            )

        fused = self._call(
            lambda: self._client.query_points(
                self._collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=self._fusion),
                query_filter=doc_filter,
                limit=limit,
                with_payload=True,
            ),
        )
        if not with_diagnostics:
            response = await fused
            return HybridSearchResult(candidates=self._to_scored(response.points))

        response, dense_ranking, sparse_ranking = await asyncio.gather(
            fused,
            self._branch_ranking(dense, DENSE_VECTOR, self._dense_limit, doc_filter),
            self._branch_ranking(sparse_query, SPARSE_VECTOR, self._sparse_limit, doc_filter)
            if sparse.indices
            else _empty(),
        )
        return HybridSearchResult(
            candidates=self._to_scored(response.points),
            dense_ranking=dense_ranking,
            sparse_ranking=sparse_ranking,
        )

    async def _branch_ranking(
        self,
        query: list[float] | models.SparseVector,
        using: str,
        limit: int,
        doc_filter: models.Filter,
    ) -> list[int]:
        response = await self._call(
            lambda: self._client.query_points(
                self._collection,
                query=query,
                using=using,
                query_filter=doc_filter,
                limit=limit,
                with_payload=["chunk_index"],
            ),
        )
        return [(point.payload or {})["chunk_index"] for point in response.points]

    async def _call[T](self, fn: Callable[[], Awaitable[T]]) -> T:
        return await self._retry.call(_SERVICE, fn, is_transient)

    @staticmethod
    def _payload(document: Document, chunk: Chunk) -> dict[str, Any]:
        return {
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "filename": document.filename,
            "chunks_total": document.chunks_total,
            "pages_total": document.pages_total,
            "indexed_at": document.indexed_at.isoformat(),
            "chunk_index": chunk.chunk_index,
            "page_from": chunk.page_from,
            "page_to": chunk.page_to,
            "section": chunk.section,
            "text": chunk.text,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "ready": False,
            "page_marks": [
                {"page": m.page, "char_start": m.char_start, "char_end": m.char_end} for m in chunk.page_marks
            ],
            "heading_marks": [{"title": m.title, "char_start": m.char_start} for m in chunk.heading_marks],
        }

    @staticmethod
    def _to_document(payload: dict[str, Any]) -> Document:
        return Document(
            doc_id=payload["doc_id"],
            doc_title=payload["doc_title"],
            filename=payload.get("filename", ""),
            chunks_total=payload.get("chunks_total", 0),
            pages_total=payload.get("pages_total"),
            indexed_at=datetime.fromisoformat(payload["indexed_at"]),
        )

    @staticmethod
    def _to_scored(points: list[models.ScoredPoint]) -> list[ScoredChunk]:
        scored = []
        for point in points:
            payload = point.payload or {}
            chunk = Chunk(
                doc_id=payload["doc_id"],
                doc_title=payload["doc_title"],
                chunk_index=payload["chunk_index"],
                text=payload["text"],
                char_start=payload["char_start"],
                char_end=payload["char_end"],
                page_from=payload.get("page_from"),
                page_to=payload.get("page_to"),
                section=payload.get("section"),
                page_marks=[
                    PageSpan(page=m["page"], char_start=m["char_start"], char_end=m["char_end"])
                    for m in payload.get("page_marks", [])
                ],
                heading_marks=[
                    Heading(char_start=m["char_start"], title=m["title"]) for m in payload.get("heading_marks", [])
                ],
            )
            scored.append(ScoredChunk(chunk=chunk, score=point.score))
        return scored


async def _empty() -> list[int]:
    return []
