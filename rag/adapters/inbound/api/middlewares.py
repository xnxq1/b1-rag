from http import HTTPStatus

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from rag.domain.exceptions import FileTooLargeError

MULTIPART_OVERHEAD_BYTES = 64 * 1024


class UploadSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, path: str, max_bytes: int) -> None:
        self._app = app
        self._path = path
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._exceeds_limit(scope):
            response = JSONResponse(
                status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                content={"detail": FileTooLargeError.for_limit(self._max_bytes).message},
                headers={"Connection": "close"},
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)

    def _exceeds_limit(self, scope: Scope) -> bool:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"].rstrip("/") != self._path:
            return False
        content_length = Headers(scope=scope).get("content-length", "")
        return content_length.isdigit() and int(content_length) > self._max_bytes + MULTIPART_OVERHEAD_BYTES
