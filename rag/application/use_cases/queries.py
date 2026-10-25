from rag.application.ports import TaskStatusRepository, VectorStorePort
from rag.domain.entities import Document, IndexingTask
from rag.domain.exceptions import TaskNotFoundError


class GetIndexingStatus:
    def __init__(self, statuses: TaskStatusRepository) -> None:
        self._statuses = statuses

    async def execute(self, task_id: str) -> IndexingTask:
        task = await self._statuses.get(task_id)
        if task is None:
            msg = f"Задача {task_id} не найдена"
            raise TaskNotFoundError(msg)
        return task


class ListDocuments:
    def __init__(self, vector_store: VectorStorePort) -> None:
        self._vector_store = vector_store

    async def execute(self) -> list[Document]:
        documents = await self._vector_store.list_documents()
        return sorted(documents, key=lambda d: d.indexed_at, reverse=True)
