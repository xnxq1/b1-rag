import asyncio

from celery import Celery
from kombu.exceptions import OperationalError

from rag.adapters.outbound.tasks.celery_app import INDEX_DOCUMENT_TASK
from rag.domain.exceptions import ExternalServiceUnavailableError


class CeleryTaskQueue:
    def __init__(self, app: Celery) -> None:
        self._app = app

    async def enqueue_indexing(self, task_id: str) -> None:
        try:
            await asyncio.to_thread(
                self._app.send_task,
                INDEX_DOCUMENT_TASK,
                kwargs={"task_id": task_id},
                task_id=task_id,
                retry=True,
            )
        except OperationalError as exc:
            msg = "Брокер задач недоступен, попробуйте позже"
            raise ExternalServiceUnavailableError(msg) from exc
