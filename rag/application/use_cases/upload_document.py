from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rag.application.ports import FileStoragePort, TaskQueuePort, TaskStatusRepository
from rag.domain.entities import IndexingStage, IndexingTask, TaskStatus
from rag.domain.exceptions import EmptyDocumentError, UnsupportedFileTypeError


def stored_file_name(task: IndexingTask) -> str:
    return f"{task.doc_id}{Path(task.filename).suffix.lower()}"


class UploadDocument:
    def __init__(
        self,
        storage: FileStoragePort,
        statuses: TaskStatusRepository,
        queue: TaskQueuePort,
        supported_extensions: frozenset[str],
        max_bytes: int,
    ) -> None:
        self._storage = storage
        self._statuses = statuses
        self._queue = queue
        self._supported_extensions = supported_extensions
        self._max_bytes = max_bytes

    async def execute(self, filename: str, content: AsyncIterator[bytes], title: str | None = None) -> IndexingTask:
        extension = Path(filename).suffix.lower()
        if extension not in self._supported_extensions:
            allowed = ", ".join(sorted(self._supported_extensions))
            msg = f"Неподдерживаемый формат файла «{extension or filename}». Допустимы: {allowed}"
            raise UnsupportedFileTypeError(msg)

        doc_id = str(uuid4())
        now = datetime.now(UTC)
        task = IndexingTask(
            task_id=doc_id,
            doc_id=doc_id,
            doc_title=(title or "").strip() or Path(filename).stem,
            filename=Path(filename).name,
            status=TaskStatus.QUEUED,
            stage=IndexingStage.QUEUED,
            progress=0.0,
            created_at=now,
            updated_at=now,
        )

        size = await self._storage.save(stored_file_name(task), content, self._max_bytes)
        if size == 0:
            await self._storage.delete(stored_file_name(task))
            msg = "Загружен пустой файл"
            raise EmptyDocumentError(msg)

        await self._statuses.save(task)
        try:
            await self._queue.enqueue_indexing(task.task_id)
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = f"Не удалось поставить задачу в очередь: {exc}"
            task.updated_at = datetime.now(UTC)
            await self._statuses.save(task)
            raise
        return task
