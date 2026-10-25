from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from rag.adapters.inbound.api.errors import register_exception_handlers
from rag.adapters.inbound.api.middlewares import UploadSizeLimitMiddleware
from rag.adapters.inbound.api.routers.base import BaseRouter
from rag.application.ports import VectorStorePort


class AppBuilder:
    def __init__(
        self,
        routers: list[BaseRouter],
        vector_store: VectorStorePort,
        max_upload_bytes: int,
        api_prefix: str = "/api",
    ) -> None:
        self._routers = routers
        self._vector_store = vector_store
        self._max_upload_bytes = max_upload_bytes
        self._api_prefix = api_prefix

    @asynccontextmanager
    async def lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        await self._vector_store.ensure_collection()
        yield

    def create_app(self) -> FastAPI:
        app = FastAPI(
            title="b1-rag",
            description="Вопросы по документу с ответами строго по тексту и ссылками на фрагменты",
            version="0.1.0",
            lifespan=self.lifespan,
        )
        register_exception_handlers(app)
        app.add_middleware(
            UploadSizeLimitMiddleware,
            path=f"{self._api_prefix}/documents",
            max_bytes=self._max_upload_bytes,
        )
        api_router = APIRouter(prefix=self._api_prefix)
        for r in self._routers:
            api_router.include_router(r.router)
        app.include_router(api_router)
        return app
