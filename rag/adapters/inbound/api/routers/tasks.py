from typing import ClassVar

from fastapi import APIRouter

from rag.adapters.inbound.api.routers.base import BaseRouter
from rag.adapters.inbound.api.schemas import ErrorResponse, TaskResponse
from rag.application.use_cases import GetIndexingStatus


class TasksRouter(BaseRouter):
    prefix = "/tasks"
    tags: ClassVar[list[str]] = ["tasks"]

    def __init__(self, get_indexing_status: GetIndexingStatus) -> None:
        self._get_indexing_status = get_indexing_status
        super().__init__()

    def _setup_router_endpoints(self, router: APIRouter) -> None:
        @router.get(
            "/{task_id}",
            name="get_task",
            responses={404: {"model": ErrorResponse}},
            summary="Состояние фоновой задачи индексации",
        )
        async def get_task(task_id: str) -> TaskResponse:
            return TaskResponse.from_domain(await self._get_indexing_status.execute(task_id))
