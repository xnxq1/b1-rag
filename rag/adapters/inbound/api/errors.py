import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag.domain.exceptions import (
    DocumentNotFoundError,
    DomainError,
    EmptyDocumentError,
    ExternalServiceUnavailableError,
    FileTooLargeError,
    InvalidModelResponseError,
    TaskNotFoundError,
    TooManyPagesError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

DOMAIN_ERROR_STATUSES: dict[type[DomainError], HTTPStatus] = {
    UnsupportedFileTypeError: HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
    FileTooLargeError: HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
    EmptyDocumentError: HTTPStatus.UNPROCESSABLE_ENTITY,
    TooManyPagesError: HTTPStatus.UNPROCESSABLE_ENTITY,
    DocumentNotFoundError: HTTPStatus.NOT_FOUND,
    TaskNotFoundError: HTTPStatus.NOT_FOUND,
    ExternalServiceUnavailableError: HTTPStatus.SERVICE_UNAVAILABLE,
    InvalidModelResponseError: HTTPStatus.BAD_GATEWAY,
}


def status_for(exc: DomainError) -> HTTPStatus:
    for cls in type(exc).__mro__:
        if cls in DOMAIN_ERROR_STATUSES:
            return DOMAIN_ERROR_STATUSES[cls]
    return HTTPStatus.BAD_REQUEST


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _domain_handler(_: Request, exc: DomainError) -> JSONResponse:
        status = status_for(exc)
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
            logger.warning("%s: %s", type(exc).__name__, exc.message, exc_info=exc.__cause__)
        return JSONResponse(status_code=status, content={"detail": exc.message})

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )
