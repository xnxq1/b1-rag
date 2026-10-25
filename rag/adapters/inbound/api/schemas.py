from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field

from rag.application.dto import AskResult, RankedFragment, SearchDiagnostics
from rag.domain.entities import Citation, Document, IndexingStage, IndexingTask, TaskStatus


class UploadResponse(BaseModel):
    doc_id: str
    task_id: str
    status: TaskStatus
    status_url: str


class TaskResponse(BaseModel):
    task_id: str
    doc_id: str
    doc_title: str
    filename: str
    status: TaskStatus
    stage: IndexingStage
    progress: float = Field(description="Доля выполнения, 0..1")
    chunks_total: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, task: IndexingTask) -> Self:
        return cls(
            task_id=task.task_id,
            doc_id=task.doc_id,
            doc_title=task.doc_title,
            filename=task.filename,
            status=task.status,
            stage=task.stage,
            progress=task.progress,
            chunks_total=task.chunks_total,
            error=task.error,
            created_at=task.created_at,
            updated_at=task.updated_at,
        )


class DocumentResponse(BaseModel):
    doc_id: str
    doc_title: str
    filename: str
    chunks_total: int
    pages_total: int | None
    indexed_at: datetime

    @classmethod
    def from_domain(cls, document: Document) -> Self:
        return cls(
            doc_id=document.doc_id,
            doc_title=document.doc_title,
            filename=document.filename,
            chunks_total=document.chunks_total,
            pages_total=document.pages_total,
            indexed_at=document.indexed_at,
        )


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["Какой штраф за просрочку поставки?"])


class CitationResponse(BaseModel):
    text: str = Field(description="Дословный фрагмент текста документа")
    page: int | None = Field(description="Страница начала фрагмента; null, если формат не даёт страниц")
    page_to: int | None
    section: str | None
    chunk_index: int
    char_start: int = Field(description="Смещение цитаты в полном тексте документа")
    char_end: int

    @classmethod
    def from_domain(cls, citation: Citation) -> Self:
        return cls(
            text=citation.text,
            page=citation.page,
            page_to=citation.page_to,
            section=citation.section,
            chunk_index=citation.chunk_index,
            char_start=citation.char_start,
            char_end=citation.char_end,
        )


class RankedFragmentResponse(BaseModel):
    chunk_index: int
    page_from: int | None
    page_to: int | None
    section: str | None
    rerank_score: float
    dense_rank: int | None = Field(description="Место фрагмента в плотной ветви; null — ветвь его не нашла")
    sparse_rank: int | None = Field(description="Место фрагмента в ветви BM25; null — ветвь его не нашла")
    used_in_context: bool

    @classmethod
    def from_dto(cls, fragment: RankedFragment) -> Self:
        return cls(
            chunk_index=fragment.chunk_index,
            page_from=fragment.page_from,
            page_to=fragment.page_to,
            section=fragment.section,
            rerank_score=fragment.rerank_score,
            dense_rank=fragment.dense_rank,
            sparse_rank=fragment.sparse_rank,
            used_in_context=fragment.used_in_context,
        )


class DiagnosticsResponse(BaseModel):
    dense_hits: int = Field(description="Сколько фрагментов дала плотная (семантическая) ветвь")
    sparse_hits: int = Field(description="Сколько фрагментов дала разреженная ветвь BM25")
    fused_candidates: int = Field(description="Кандидатов после слияния ветвей (вход реранкера)")
    reranked_candidates: int = Field(description="Кандидатов после переранжирования")
    context_fragments: int = Field(description="Фрагментов, переданных LLM (после порога релевантности)")
    used_in_answer: list[int] = Field(description="chunk_index фрагментов, на которые опирается ответ")
    llm_called: bool
    reranked: list[RankedFragmentResponse]

    @classmethod
    def from_dto(cls, diagnostics: SearchDiagnostics) -> Self:
        return cls(
            dense_hits=diagnostics.dense_hits,
            sparse_hits=diagnostics.sparse_hits,
            fused_candidates=diagnostics.fused_candidates,
            reranked_candidates=len(diagnostics.reranked),
            context_fragments=diagnostics.context_fragments,
            used_in_answer=diagnostics.used_in_answer,
            llm_called=diagnostics.llm_called,
            reranked=[RankedFragmentResponse.from_dto(f) for f in diagnostics.reranked],
        )


class AnswerResponse(BaseModel):
    answer: str
    found: bool
    citations: list[CitationResponse]
    diagnostics: DiagnosticsResponse | None = Field(default=None, description="Только при debug=true")

    @classmethod
    def from_result(cls, result: AskResult) -> Self:
        return cls(
            answer=result.answer.answer,
            found=result.answer.found,
            citations=[CitationResponse.from_domain(c) for c in result.answer.citations],
            diagnostics=DiagnosticsResponse.from_dto(result.diagnostics) if result.diagnostics else None,
        )


class ErrorResponse(BaseModel):
    detail: str
