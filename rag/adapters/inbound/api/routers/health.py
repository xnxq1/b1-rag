from typing import ClassVar

from fastapi import APIRouter

from rag.adapters.inbound.api.routers.base import BaseRouter


class HealthRouter(BaseRouter):
    tags: ClassVar[list[str]] = ["health"]

    def _setup_router_endpoints(self, router: APIRouter) -> None:
        @router.get("/health")
        async def health() -> dict[str, str]:
            # TODO: тут по хорошему должен быть список, зависимых компонентов в формате {'postgresql': true, 'redis': false} и т.д.
            # это позволит по логам определять что отключалось из компонентов + сделать рестарт
            return {"status": "ok"}
