import asyncio

from celery.signals import worker_process_shutdown

from rag.adapters.outbound.tasks.celery_app import INDEX_DOCUMENT_TASK
from rag.application.ports import VectorStorePort
from rag.application.use_cases import IndexDocument
from rag.di import celery_app, container


class _Runtime:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._index_document: IndexDocument | None = None

    def index_document(self, task_id: str) -> None:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._index(task_id))

    async def _index(self, task_id: str) -> None:
        if self._index_document is None:
            vector_store = await container.get(VectorStorePort)
            await vector_store.ensure_collection()
            self._index_document = await container.get(IndexDocument)
        await self._index_document.execute(task_id)

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


@celery_app.task(name=INDEX_DOCUMENT_TASK)
def index_document(task_id: str) -> None:
    _runtime.index_document(task_id)
