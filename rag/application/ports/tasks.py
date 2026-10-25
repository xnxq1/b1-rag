from typing import Protocol

from rag.domain.entities import IndexingTask


class TaskStatusRepository(Protocol):
    async def save(self, task: IndexingTask) -> None: ...

    async def get(self, task_id: str) -> IndexingTask | None: ...


class TaskQueuePort(Protocol):
    async def enqueue_indexing(self, task_id: str) -> None: ...
