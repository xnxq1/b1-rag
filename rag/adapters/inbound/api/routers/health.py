import asyncio
import logging
from http import HTTPStatus
from typing import ClassVar, Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel

from rag.adapters.inbound.api.routers.base import BaseRouter
from rag.application.ports.health import DependencyCheckPort

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    components: dict[str, bool]


class HealthRouter(BaseRouter):
    tags: ClassVar[list[str]] = ["health"]

    def __init__(self, checks: list[DependencyCheckPort], timeout_seconds: float = 2.0) -> None:
        self._checks = checks
        self._timeout = timeout_seconds
        super().__init__()

    def _setup_router_endpoints(self, router: APIRouter) -> None:
        @router.get(
            "/health",
            responses={503: {"model": HealthResponse}},
            summary="Доступность зависимостей: Qdrant, Redis",
        )
        async def health(response: Response) -> HealthResponse:
            results = await asyncio.gather(*(self._check(c) for c in self._checks))
            components = {check.name: ok for check, ok in zip(self._checks, results, strict=True)}
            if all(components.values()):
                return HealthResponse(status="ok", components=components)
            response.status_code = HTTPStatus.SERVICE_UNAVAILABLE
            return HealthResponse(status="unavailable", components=components)

    async def _check(self, check: DependencyCheckPort) -> bool:
        try:
            return await asyncio.wait_for(check.is_available(), timeout=self._timeout)
        except Exception as exc:
            logger.warning("Компонент %s недоступен: %r", check.name, exc)
            return False
