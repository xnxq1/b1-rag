from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class IndexingStage(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    UPSERTING = "upserting"
    DONE = "done"


@dataclass(slots=True)
class IndexingTask:
    task_id: str
    doc_id: str
    doc_title: str
    filename: str
    status: TaskStatus
    stage: IndexingStage
    progress: float
    created_at: datetime
    updated_at: datetime
    chunks_total: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PageSpan:
    page: int
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class Heading:
    char_start: int
    title: str


@dataclass(slots=True)
class ParsedDocument:
    text: str
    pages: list[PageSpan] = field(default_factory=list)
    headings: list[Heading] = field(default_factory=list)
    pages_total: int | None = None


@dataclass(frozen=True, slots=True)
class TextSpan:
    text: str
    char_start: int
    char_end: int


@dataclass(slots=True)
class Chunk:
    doc_id: str
    doc_title: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_from: int | None
    page_to: int | None
    section: str | None
    page_marks: list[PageSpan] = field(default_factory=list)
    heading_marks: list[Heading] = field(default_factory=list)

    def page_at(self, offset: int) -> int | None:
        page = self.page_from
        for mark in self.page_marks:
            if mark.char_start <= offset:
                page = mark.page
        return page

    def section_at(self, offset: int) -> str | None:
        section = self.section
        for mark in self.heading_marks:
            if mark.char_start <= offset:
                section = mark.title
        return section


@dataclass(slots=True)
class Document:
    doc_id: str
    doc_title: str
    filename: str
    chunks_total: int
    pages_total: int | None
    indexed_at: datetime


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class DraftCitation:
    fragment: int
    quote: str


@dataclass(slots=True)
class AnswerDraft:
    found: bool
    answer: str
    citations: list[DraftCitation]


@dataclass(frozen=True, slots=True)
class Citation:
    text: str
    page: int | None
    page_to: int | None
    section: str | None
    chunk_index: int
    char_start: int
    char_end: int


@dataclass(slots=True)
class Answer:
    answer: str
    found: bool
    citations: list[Citation]
