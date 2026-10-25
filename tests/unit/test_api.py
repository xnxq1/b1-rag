from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from rag.adapters.inbound.api.builder import AppBuilder
from rag.adapters.inbound.api.errors import DOMAIN_ERROR_STATUSES
from rag.adapters.inbound.api.routers.documents import DocumentsRouter
from rag.adapters.inbound.api.routers.health import HealthRouter
from rag.adapters.inbound.api.routers.tasks import TasksRouter
from rag.application.use_cases import AskQuestion, GetIndexingStatus, ListDocuments, UploadDocument
from rag.domain import exceptions
from rag.domain.entities import AnswerDraft, DraftCitation
from rag.domain.grounding import NOT_FOUND_MESSAGE
from tests.fakes import (
    DOC_ID,
    FakeEmbeddings,
    FakeGenerator,
    FakeQueue,
    FakeReranker,
    FakeSparse,
    FakeStatuses,
    FakeStorage,
    FakeVectorStore,
    make_chunk,
)

PENALTY = "За нарушение сроков поставки Поставщик уплачивает неустойку в размере 0,1% за каждый день."


@pytest.fixture
def storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def queue() -> FakeQueue:
    return FakeQueue()


@pytest.fixture
async def client(storage: FakeStorage, queue: FakeQueue) -> AsyncIterator[AsyncClient]:
    statuses = FakeStatuses()
    store = FakeVectorStore([make_chunk(0, PENALTY, page=12, section="4.2 Штрафные санкции")])
    generator = FakeGenerator(AnswerDraft(True, "0,1% в день.", [DraftCitation(1, "неустойку в размере 0,1%")]))
    ask = AskQuestion(
        FakeEmbeddings(),
        FakeSparse(),
        store,
        FakeReranker("неустойк", query_word="штраф"),
        generator,
        rerank_candidates=10,
        rerank_top_n=3,
        rerank_min_score=0.1,
    )
    routers = [
        DocumentsRouter(
            upload_document=UploadDocument(storage, statuses, queue, frozenset({".pdf", ".docx"}), 1024),
            list_documents=ListDocuments(store),
            ask_question=ask,
        ),
        TasksRouter(get_indexing_status=GetIndexingStatus(statuses)),
        HealthRouter(checks=[]),
    ]
    app = AppBuilder(routers=routers, vector_store=store, max_upload_bytes=1024).create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_upload_and_status(client: AsyncClient) -> None:
    response = await client.post(
        "/api/documents",
        files={"file": ("Договор.pdf", b"%PDF-1.7", "application/pdf")},
        data={"title": "Договор поставки"},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["status_url"].endswith(f"/api/tasks/{body['task_id']}")

    status = await client.get(f"/api/tasks/{body['task_id']}")
    assert status.status_code == 200
    assert status.json()["doc_title"] == "Договор поставки"


async def test_upload_errors(client: AsyncClient) -> None:
    unsupported = await client.post("/api/documents", files={"file": ("a.txt", b"x", "text/plain")})
    assert unsupported.status_code == 415
    too_large = await client.post("/api/documents", files={"file": ("a.pdf", b"x" * 2048, "application/pdf")})
    assert too_large.status_code == 413
    assert (await client.get(f"/api/tasks/{uuid4()}")).status_code == 404
    assert (await client.get("/api/tasks/unknown")).status_code == 422


async def test_ask_with_debug(client: AsyncClient) -> None:
    response = await client.post(f"/api/documents/{DOC_ID}/ask?debug=true", json={"question": "Какой штраф?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["citations"][0] == {
        "text": "неустойку в размере 0,1%",
        "page": 12,
        "page_to": 12,
        "section": "4.2 Штрафные санкции",
        "chunk_index": 0,
        "char_start": PENALTY.index("неустойку"),
        "char_end": PENALTY.index("неустойку") + len("неустойку в размере 0,1%"),
    }
    assert body["diagnostics"]["reranked_candidates"] == 1
    assert body["diagnostics"]["used_in_answer"] == [0]


async def test_ask_not_found_and_validation(client: AsyncClient) -> None:
    response = await client.post(f"/api/documents/{DOC_ID}/ask", json={"question": "Кто директор?"})
    assert response.json() == {"answer": NOT_FOUND_MESSAGE, "found": False, "citations": [], "diagnostics": None}
    assert (await client.post(f"/api/documents/{uuid4()}/ask", json={"question": "?"})).status_code == 404
    assert (await client.post("/api/documents/doc-1/ask", json={"question": "?"})).status_code == 422
    assert (await client.post(f"/api/documents/{DOC_ID}/ask", json={"question": ""})).status_code == 422


async def test_upload_rejected_by_content_length_before_reading_body(
    client: AsyncClient,
    storage: FakeStorage,
    queue: FakeQueue,
) -> None:
    response = await client.post("/api/documents", files={"file": ("big.pdf", b"x" * 200_000, "application/pdf")})

    assert response.status_code == 413
    assert response.json() == {"detail": "Файл больше допустимых 1 КБ"}
    assert storage.files == {}
    assert queue.enqueued == []


async def test_content_length_limit_applies_only_to_upload(client: AsyncClient) -> None:
    response = await client.post(f"/api/documents/{DOC_ID}/ask", json={"question": "Какой штраф? " * 100})
    assert response.status_code == 200


def test_every_domain_error_has_http_status() -> None:
    domain_errors = {
        cls for cls in vars(exceptions).values() if isinstance(cls, type) and issubclass(cls, exceptions.DomainError)
    }
    assert domain_errors - {exceptions.DomainError} == set(DOMAIN_ERROR_STATUSES)


class StubCheck:
    def __init__(self, name: str, result: bool | Exception) -> None:
        self.name = name
        self._result = result

    async def is_available(self) -> bool:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


async def health_client(*checks: StubCheck) -> AsyncClient:
    app = AppBuilder(
        routers=[HealthRouter(checks=list(checks), timeout_seconds=0.5)],
        vector_store=FakeVectorStore(),
        max_upload_bytes=1024,
    ).create_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_ok() -> None:
    async with await health_client(StubCheck("qdrant", True), StubCheck("redis", True)) as c:
        response = await c.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "components": {"qdrant": True, "redis": True}}


async def test_health_reports_unavailable_component() -> None:
    async with await health_client(StubCheck("qdrant", True), StubCheck("redis", ConnectionError("down"))) as c:
        response = await c.get("/api/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable", "components": {"qdrant": True, "redis": False}}
