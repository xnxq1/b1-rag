import pytest

from rag.application.use_cases import AskQuestion, GetIndexingStatus, IndexDocument, UploadDocument
from rag.domain.entities import AnswerDraft, DraftCitation, Heading, PageSpan, ParsedDocument, TaskStatus
from rag.domain.exceptions import (
    DocumentNotFoundError,
    ExternalServiceUnavailableError,
    FileTooLargeError,
    TaskNotFoundError,
    UnsupportedFileTypeError,
)
from tests.fakes import (
    FakeChunker,
    FakeEmbeddings,
    FakeGenerator,
    FakeParser,
    FakeQueue,
    FakeReranker,
    FakeSparse,
    FakeStatuses,
    FakeStorage,
    FakeVectorStore,
    bytes_stream,
    make_chunk,
)

PENALTY = "За нарушение сроков поставки Поставщик уплачивает неустойку в размере 0,1% за каждый день."


def ask_use_case(store: FakeVectorStore, generator: FakeGenerator) -> AskQuestion:
    return AskQuestion(
        embeddings=FakeEmbeddings(),
        sparse=FakeSparse(),
        vector_store=store,
        reranker=FakeReranker(keyword="неустойк"),
        generator=generator,
        rerank_candidates=10,
        rerank_top_n=3,
        rerank_min_score=0.1,
    )


async def test_ask_returns_grounded_answer_with_diagnostics() -> None:
    store = FakeVectorStore([make_chunk(0, "Общие положения."), make_chunk(1, PENALTY, page=12, section="4.2")])
    generator = FakeGenerator(AnswerDraft(True, "0,1% в день.", [DraftCitation(1, "неустойку в размере 0,1%")]))

    result = await ask_use_case(store, generator).execute("doc-1", "Какой штраф?", debug=True)

    assert result.answer.found
    assert result.answer.citations[0].page == 12
    assert [c.chunk.chunk_index for c in generator.calls[0][1]] == [1]
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert (diagnostics.dense_hits, diagnostics.sparse_hits, diagnostics.fused_candidates) == (2, 1, 2)
    assert diagnostics.context_fragments == 1
    assert diagnostics.used_in_answer == [1]
    assert diagnostics.llm_called


async def test_ask_without_relevant_fragments_does_not_call_llm() -> None:
    store = FakeVectorStore([make_chunk(0, "Общие положения."), make_chunk(1, "Порядок приемки.")])
    generator = FakeGenerator()

    result = await ask_use_case(store, generator).execute("doc-1", "Кто генеральный директор?")

    assert not result.answer.found
    assert result.answer.citations == []
    assert generator.calls == []
    assert result.diagnostics is None


async def test_ask_unknown_document() -> None:
    with pytest.raises(DocumentNotFoundError):
        await ask_use_case(FakeVectorStore(), FakeGenerator()).execute("missing", "вопрос")


def upload_use_case(storage: FakeStorage, statuses: FakeStatuses, queue: FakeQueue) -> UploadDocument:
    return UploadDocument(storage, statuses, queue, frozenset({".pdf", ".docx"}), max_bytes=10)


async def test_upload_registers_task_and_enqueues() -> None:
    storage, statuses, queue = FakeStorage(), FakeStatuses(), FakeQueue()

    task = await upload_use_case(storage, statuses, queue).execute("Договор.PDF", bytes_stream(b"%PDF"))

    assert task.status == TaskStatus.QUEUED
    assert task.doc_title == "Договор"
    assert queue.enqueued == [task.task_id]
    assert f"{task.doc_id}.pdf" in storage.files
    assert (await GetIndexingStatus(statuses).execute(task.task_id)).doc_id == task.doc_id


async def test_upload_rejects_unsupported_and_large_files() -> None:
    use_case = upload_use_case(FakeStorage(), FakeStatuses(), FakeQueue())
    with pytest.raises(UnsupportedFileTypeError):
        await use_case.execute("notes.txt", bytes_stream(b"x"))
    with pytest.raises(FileTooLargeError):
        await use_case.execute("big.pdf", bytes_stream(b"x" * 8, b"x" * 8))


async def test_unknown_task_status() -> None:
    with pytest.raises(TaskNotFoundError):
        await GetIndexingStatus(FakeStatuses()).execute("nope")


def parsed_document() -> ParsedDocument:
    text = "1. ПРЕДМЕТ\n" + "Текст первой страницы. " * 5 + "\n\n4.2 Штрафные санкции\n" + PENALTY
    second_page = text.index("4.2")
    return ParsedDocument(
        text=text,
        pages=[PageSpan(1, 0, second_page - 2), PageSpan(2, second_page, len(text))],
        headings=[Heading(0, "1. ПРЕДМЕТ"), Heading(second_page, "4.2 Штрафные санкции")],
        pages_total=2,
    )


async def make_index_case(store: FakeVectorStore) -> tuple[IndexDocument, FakeStatuses, str, FakeStorage]:
    storage, statuses, queue = FakeStorage(), FakeStatuses(), FakeQueue()
    task = await upload_use_case(storage, statuses, queue).execute("Договор.pdf", bytes_stream(b"%PDF"))
    use_case = IndexDocument(
        statuses=statuses,
        storage=storage,
        parser=FakeParser(parsed_document()),
        chunker=FakeChunker(size=60),
        embeddings=FakeEmbeddings(),
        sparse=FakeSparse(),
        vector_store=store,
        embedding_batch_size=2,
    )
    return use_case, statuses, task.task_id, storage


async def test_index_document_writes_chunks_with_layout() -> None:
    store = FakeVectorStore()
    use_case, statuses, task_id, storage = await make_index_case(store)

    await use_case.execute(task_id)

    task = statuses.tasks[task_id]
    assert task.status == TaskStatus.DONE
    assert task.progress == 1.0
    chunks = store.chunks[task_id]
    assert task.chunks_total == len(chunks) == store.documents[task_id].chunks_total
    last = chunks[-1]
    assert (last.page_to, last.section) == (2, "4.2 Штрафные санкции")
    assert storage.files == {}
    stages = [stage for _, stage in statuses.history]
    assert stages.index("parsing") < stages.index("embedding") < stages.index("upserting") < stages.index("done")


async def test_index_failure_marks_task_failed_and_cleans_up() -> None:
    store = FakeVectorStore()
    store.fail_upsert = RuntimeError("qdrant down")
    use_case, statuses, task_id, storage = await make_index_case(store)

    with pytest.raises(RuntimeError):
        await use_case.execute(task_id)

    task = statuses.tasks[task_id]
    assert task.status == TaskStatus.FAILED
    assert "qdrant down" in (task.error or "")
    assert store.deleted == [task_id]
    assert storage.files == {}


@pytest.mark.parametrize(
    ("max_bytes", "message"),
    [
        (50 * 1024 * 1024, "Файл больше допустимых 50 МБ"),
        (int(1.5 * 1024 * 1024), "Файл больше допустимых 1.5 МБ"),
        (1024, "Файл больше допустимых 1 КБ"),
        (500, "Файл больше допустимых 500 байт"),
    ],
)
def test_file_too_large_message(max_bytes: int, message: str) -> None:
    assert FileTooLargeError.for_limit(max_bytes).message == message


async def test_transient_failure_before_last_attempt_requeues_task() -> None:
    store = FakeVectorStore()
    store.fail_upsert = ExternalServiceUnavailableError("Сервис эмбеддингов временно недоступен")
    use_case, statuses, task_id, storage = await make_index_case(store)

    with pytest.raises(ExternalServiceUnavailableError):
        await use_case.execute(task_id, last_attempt=False)

    task = statuses.tasks[task_id]
    assert (task.status, task.stage, task.progress) == (TaskStatus.QUEUED, "queued", 0.0)
    assert task.error == "Временная ошибка, задача будет повторена: Сервис эмбеддингов временно недоступен"
    assert store.deleted == [task_id]
    assert len(storage.files) == 1

    store.fail_upsert = None
    await use_case.execute(task_id, last_attempt=True)
    assert statuses.tasks[task_id].status == TaskStatus.DONE
    assert statuses.tasks[task_id].error is None
    assert storage.files == {}


async def test_transient_failure_on_last_attempt_fails_task() -> None:
    store = FakeVectorStore()
    store.fail_upsert = ExternalServiceUnavailableError("Сервис эмбеддингов временно недоступен")
    use_case, statuses, task_id, storage = await make_index_case(store)

    with pytest.raises(ExternalServiceUnavailableError):
        await use_case.execute(task_id, last_attempt=True)

    assert statuses.tasks[task_id].status == TaskStatus.FAILED
    assert storage.files == {}
