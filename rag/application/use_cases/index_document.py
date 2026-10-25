import logging
from datetime import UTC, datetime

from rag.application.ports import (
    ChunkerPort,
    DocumentParserPort,
    EmbeddingsPort,
    FileStoragePort,
    SparseEncoderPort,
    TaskStatusRepository,
    VectorStorePort,
)
from rag.application.use_cases.upload_document import stored_file_name
from rag.domain.entities import Document, IndexingStage, IndexingTask, TaskStatus
from rag.domain.exceptions import DomainError, EmptyDocumentError, TaskNotFoundError
from rag.domain.layout import build_chunks

logger = logging.getLogger(__name__)

_EMBEDDING_START = 0.2
_EMBEDDING_END = 0.85


class IndexDocument:
    def __init__(
        self,
        statuses: TaskStatusRepository,
        storage: FileStoragePort,
        parser: DocumentParserPort,
        chunker: ChunkerPort,
        embeddings: EmbeddingsPort,
        sparse: SparseEncoderPort,
        vector_store: VectorStorePort,
        embedding_batch_size: int,
    ) -> None:
        self._statuses = statuses
        self._storage = storage
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._sparse = sparse
        self._vector_store = vector_store
        self._batch_size = embedding_batch_size

    async def execute(self, task_id: str) -> None:
        task = await self._statuses.get(task_id)
        if task is None:
            msg = f"Задача {task_id} не найдена"
            raise TaskNotFoundError(msg)
        if task.status == TaskStatus.DONE:
            return

        try:
            await self._index(task)
        except Exception as exc:
            logger.exception("Индексация документа %s завершилась ошибкой", task.doc_id)
            await self._cleanup(task.doc_id)
            task.status = TaskStatus.FAILED
            task.error = exc.message if isinstance(exc, DomainError) else f"{type(exc).__name__}: {exc}"[:500]
            await self._save(task)
            raise

    async def _index(self, task: IndexingTask) -> None:
        task.status = TaskStatus.RUNNING
        task.error = None
        await self._advance(task, IndexingStage.PARSING, 0.05)
        path = await self._storage.local_path(stored_file_name(task))
        parsed = await self._parser.parse(path)
        if not parsed.text.strip():
            msg = "Не удалось извлечь текст из документа (возможно, это скан без текстового слоя)"
            raise EmptyDocumentError(msg)

        await self._advance(task, IndexingStage.CHUNKING, 0.15)
        spans = self._chunker.split(parsed.text)
        chunks = build_chunks(task.doc_id, task.doc_title, parsed, spans)
        texts = [c.text for c in chunks]

        await self._advance(task, IndexingStage.EMBEDDING, _EMBEDDING_START)
        dense: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            dense.extend(await self._embeddings.embed_documents(texts[start : start + self._batch_size]))
            done = len(dense) / len(texts)
            await self._advance(
                task, IndexingStage.EMBEDDING, _EMBEDDING_START + done * (_EMBEDDING_END - _EMBEDDING_START)
            )
        sparse = await self._sparse.encode_documents(texts)

        await self._advance(task, IndexingStage.UPSERTING, 0.9)
        document = Document(
            doc_id=task.doc_id,
            doc_title=task.doc_title,
            filename=task.filename,
            chunks_total=len(chunks),
            pages_total=parsed.pages_total,
            indexed_at=datetime.now(UTC),
        )
        await self._vector_store.upsert_document(document, chunks, dense, sparse)

        task.status = TaskStatus.DONE
        task.chunks_total = len(chunks)
        await self._advance(task, IndexingStage.DONE, 1.0)
        logger.info("Документ %s проиндексирован: %d фрагментов", task.doc_id, len(chunks))

    async def _advance(self, task: IndexingTask, stage: IndexingStage, progress: float) -> None:
        task.stage = stage
        task.progress = round(progress, 3)
        await self._save(task)

    async def _save(self, task: IndexingTask) -> None:
        task.updated_at = datetime.now(UTC)
        await self._statuses.save(task)

    async def _cleanup(self, doc_id: str) -> None:
        try:
            await self._vector_store.delete_document(doc_id)
        except Exception:
            logger.exception("Не удалось удалить частично записанные фрагменты документа %s", doc_id)
