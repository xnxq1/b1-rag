import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from rag.adapters.outbound.retry import RetryPolicy
from rag.adapters.outbound.sparse.bm25 import FastEmbedBm25Encoder
from rag.adapters.outbound.vector.qdrant_store import QdrantVectorStore
from rag.domain.entities import Chunk, Document
from tests.fakes import make_chunk

pytestmark = pytest.mark.integration

DIM = 8
TEXTS = [
    "1.1. Поставщик обязуется передать Покупателю товар, а Покупатель принять и оплатить его.",
    "4.2. За нарушение сроков поставки Поставщик уплачивает Покупателю неустойку в размере 0,1% за каждый день.",
    "2.4. Покупатель оплачивает товар в течение 10 банковских дней с даты поставки.",
    "Цена договора составляет 1 500 000 рублей, включая НДС.",
]


def fake_dense(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 + 0.01 for b in digest[:DIM]]


@pytest.fixture(scope="module")
def bm25() -> FastEmbedBm25Encoder:
    return FastEmbedBm25Encoder(language="russian", avg_len=40)


@pytest.fixture
async def store(qdrant_client: AsyncQdrantClient) -> QdrantVectorStore:
    store = QdrantVectorStore(
        client=qdrant_client,
        retry=RetryPolicy(attempts=2, max_wait_seconds=0.1),
        collection=f"test_{uuid4().hex[:8]}",
        dense_dim=DIM,
        fusion="rrf",
        dense_limit=10,
        sparse_limit=10,
        upsert_batch_size=2,
    )
    await store.ensure_collection()
    await store.ensure_collection()
    return store


async def index(store: QdrantVectorStore, bm25: FastEmbedBm25Encoder, doc_id: str, texts: list[str]) -> list[Chunk]:
    chunks = [make_chunk(i, t, doc_id=doc_id, page=i + 1, section=t[:3]) for i, t in enumerate(texts)]
    document = Document(doc_id, "Договор", "contract.pdf", len(chunks), len(chunks), datetime.now(UTC))
    await store.upsert_document(document, chunks, [fake_dense(t) for t in texts], await bm25.encode_documents(texts))
    return chunks


async def test_hybrid_search_finds_exact_identifiers(store: QdrantVectorStore, bm25: FastEmbedBm25Encoder) -> None:
    await index(store, bm25, "doc-a", TEXTS)
    await index(store, bm25, "doc-b", ["4.2. Посторонний документ про неустойку 0,1%."])

    for query, expected in [("пункт 4.2 неустойка", 1), ("1 500 000", 3), ("10 банковских дней", 2)]:
        result = await store.hybrid_search(
            "doc-a",
            fake_dense(query),
            await bm25.encode_query(query),
            limit=4,
            with_diagnostics=True,
        )
        assert result.candidates[0].chunk.chunk_index == expected, query
        assert {c.chunk.doc_id for c in result.candidates} == {"doc-a"}
        assert result.dense_ranking is not None
        assert len(result.dense_ranking) == len(TEXTS)
        assert result.sparse_ranking is not None
        assert 1 <= len(result.sparse_ranking) < len(TEXTS)
        assert result.sparse_ranking[0] == expected


async def test_payload_roundtrip_list_and_delete(store: QdrantVectorStore, bm25: FastEmbedBm25Encoder) -> None:
    chunks = await index(store, bm25, "doc-a", TEXTS)
    await index(store, bm25, "doc-a", TEXTS)

    result = await store.hybrid_search("doc-a", fake_dense("x"), await bm25.encode_query("неустойка"), limit=10)
    assert len(result.candidates) == len(TEXTS)
    found = next(c.chunk for c in result.candidates if c.chunk.chunk_index == 1)
    assert found == chunks[1]

    documents = await store.list_documents()
    assert [(d.doc_id, d.chunks_total) for d in documents] == [("doc-a", len(TEXTS))]
    assert await store.document_exists("doc-a")

    await store.delete_document("doc-a")
    assert not await store.document_exists("doc-a")
    assert await store.list_documents() == []


async def test_dimension_mismatch_is_reported(qdrant_client: AsyncQdrantClient, store: QdrantVectorStore) -> None:
    other = QdrantVectorStore(
        client=qdrant_client,
        retry=RetryPolicy(attempts=1, max_wait_seconds=0.1),
        collection=store._collection,
        dense_dim=DIM * 2,
        fusion="dbsf",
        dense_limit=5,
        sparse_limit=5,
        upsert_batch_size=2,
    )
    with pytest.raises(RuntimeError, match="EMBEDDING_DIM"):
        await other.ensure_collection()


async def test_document_is_invisible_until_all_points_are_written(
    store: QdrantVectorStore,
    bm25: FastEmbedBm25Encoder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish = store._publish

    async def skip_publish(doc_id: str) -> None:
        return None

    monkeypatch.setattr(store, "_publish", skip_publish)
    await index(store, bm25, "doc-a", TEXTS)

    assert not await store.document_exists("doc-a")
    assert await store.list_documents() == []
    result = await store.hybrid_search("doc-a", fake_dense("x"), await bm25.encode_query("неустойка"), limit=10)
    assert result.candidates == []

    await publish("doc-a")

    assert await store.document_exists("doc-a")
    assert [d.doc_id for d in await store.list_documents()] == ["doc-a"]
