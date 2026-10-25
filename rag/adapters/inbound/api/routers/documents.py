from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Annotated, ClassVar

from fastapi import APIRouter, File, Form, Query, Request, UploadFile

from rag.adapters.inbound.api.routers.base import BaseRouter
from rag.adapters.inbound.api.schemas import (
    AnswerResponse,
    AskRequest,
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
    UploadResponse,
)
from rag.application.use_cases import AskQuestion, ListDocuments, UploadDocument

_READ_CHUNK = 1024 * 1024


async def _read_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    while chunk := await file.read(_READ_CHUNK):
        yield chunk


class DocumentsRouter(BaseRouter):
    prefix = "/documents"
    tags: ClassVar[list[str]] = ["documents"]

    def __init__(
        self, upload_document: UploadDocument, list_documents: ListDocuments, ask_question: AskQuestion
    ) -> None:
        self._upload_document = upload_document
        self._list_documents = list_documents
        self._ask_question = ask_question
        super().__init__()

    def _setup_router_endpoints(self, router: APIRouter) -> None:
        @router.post(
            "",
            status_code=HTTPStatus.ACCEPTED,
            responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
            summary="Загрузить документ (PDF, DOCX) и запустить индексацию",
        )
        async def upload(
            request: Request,
            file: Annotated[UploadFile, File(description="PDF или DOCX, до 100 страниц")],
            title: Annotated[str | None, Form(max_length=300)] = None,
        ) -> UploadResponse:
            task = await self._upload_document.execute(file.filename or "", _read_chunks(file), title=title)
            return UploadResponse(
                doc_id=task.doc_id,
                task_id=task.task_id,
                status=task.status,
                status_url=str(request.url_for("get_task", task_id=task.task_id)),
            )

        @router.get("", summary="Список проиндексированных документов")
        async def list_documents() -> DocumentListResponse:
            documents = await self._list_documents.execute()
            return DocumentListResponse(
                documents=[DocumentResponse.from_domain(d) for d in documents],
                total=len(documents),
            )

        @router.post(
            "/{doc_id}/ask",
            responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
            summary="Ответ на вопрос по документу со ссылками на фрагменты",
        )
        async def ask(
            doc_id: str,
            body: AskRequest,
            debug: Annotated[bool, Query(description="Добавить диагностику поиска")] = False,
        ) -> AnswerResponse:
            # TODO: тут нужно убирать debug параметр для прода
            result = await self._ask_question.execute(doc_id, body.question, debug=debug)
            return AnswerResponse.from_result(result)
