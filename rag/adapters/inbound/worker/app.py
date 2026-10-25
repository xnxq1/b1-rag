import asyncio

from celery import Task
from celery.signals import worker_process_shutdown

from rag.adapters.outbound.tasks.celery_app import INDEX_DOCUMENT_TASK
from rag.application.ports import VectorStorePort
from rag.application.use_cases import IndexDocument
from rag.di import celery_app, container, settings
from rag.domain.exceptions import ExternalServiceUnavailableError


class _Runtime:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._index_document: IndexDocument | None = None

    def index_document(self, task_id: str, last_attempt: bool) -> None:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._index(task_id, last_attempt))

    async def _index(self, task_id: str, last_attempt: bool) -> None:
        if self._index_document is None:
            vector_store = await container.get(VectorStorePort)
            await vector_store.ensure_collection()
            self._index_document = await container.get(IndexDocument)
        await self._index_document.execute(task_id, last_attempt=last_attempt)

    def close(self) -> None:
        if self._loop is None:
            return
        self._loop.run_until_complete(container.close())
        self._loop.close()
        self._loop = None


_runtime = _Runtime()


@worker_process_shutdown.connect
def _shutdown(**_: object) -> None:
    _runtime.close()


def retry_countdown(retries: int) -> int:
    backoff = settings.index_task_retry_backoff_seconds * 2**retries
    return min(backoff, settings.index_task_retry_max_backoff_seconds)


@celery_app.task(bind=True, name=INDEX_DOCUMENT_TASK, max_retries=settings.index_task_max_retries)
def index_document(self: Task, task_id: str) -> None:
    retries = self.request.retries
    try:
        _runtime.index_document(task_id, last_attempt=retries >= settings.index_task_max_retries)
    except ExternalServiceUnavailableError as exc:
        if retries >= settings.index_task_max_retries:
            raise
        raise self.retry(exc=exc, countdown=retry_countdown(retries)) from exc
