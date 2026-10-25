from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from rag.application.ports import HybridSearchResult, SparseVector
from rag.domain.entities import AnswerDraft, Chunk, Document, IndexingTask, ParsedDocument, ScoredChunk, TextSpan
from rag.domain.exceptions import FileTooLargeError


def make_chunk(index: int, text: str, doc_id: str = "doc-1", page: int | None = 1, section: str | None = None) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        doc_title="Договор поставки",
        chunk_index=index,
        text=text,
        char_start=index * 1000,
        char_end=index * 1000 + len(text),
        page_from=page,
        page_to=page,
        section=section,
    )


class FakeEmbeddings:
    dimension = 4

    def __init__(self) -> None:
        self.calls = 0

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(t)), 0.0, 0.0, 1.0] for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 0.0, 1.0]


class FakeSparse:
    async def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    async def encode_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1], values=[1.0])


class FakeVectorStore:
    def __init__(self, candidates: list[Chunk] | None = None) -> None:
        self.candidates = candidates or []
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, list[Chunk]] = {}
        self.deleted: list[str] = []
        self.fail_upsert: Exception | None = None

    async def ensure_collection(self) -> None:
        return None

    async def upsert_document(self, document: Document, chunks: list[Chunk], dense, sparse) -> None:
        if self.fail_upsert:
            raise self.fail_upsert
        assert len(chunks) == len(dense) == len(sparse)
        self.documents[document.doc_id] = document
        self.chunks[document.doc_id] = chunks

    async def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)
        self.documents.pop(doc_id, None)

    async def document_exists(self, doc_id: str) -> bool:
        return doc_id in self.documents or any(c.doc_id == doc_id for c in self.candidates)

    async def list_documents(self) -> list[Document]:
        return list(self.documents.values())

    async def hybrid_search(self, doc_id, dense, sparse, limit, with_diagnostics=False) -> HybridSearchResult:
        candidates = [ScoredChunk(c, 1.0 / (i + 1)) for i, c in enumerate(self.candidates[:limit])]
        if with_diagnostics:
            return HybridSearchResult(candidates, dense_hits=len(self.candidates), sparse_hits=1)
        return HybridSearchResult(candidates)


class FakeReranker:
    def __init__(self, keyword: str, query_word: str | None = None) -> None:
        self.keyword = keyword
        self.query_word = query_word

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        relevant_query = self.query_word is None or self.query_word in query.lower()
        scored = [
            ScoredChunk(c.chunk, 0.9 if relevant_query and self.keyword in c.chunk.text else 0.01) for c in candidates
        ]
        return sorted(scored, key=lambda c: c.score, reverse=True)[:top_n]


class FakeGenerator:
    def __init__(self, draft: AnswerDraft | None = None) -> None:
        self.draft = draft
        self.calls: list[tuple[str, list[ScoredChunk]]] = []

    async def generate(self, question: str, context: list[ScoredChunk]) -> AnswerDraft:
        self.calls.append((question, context))
        assert self.draft is not None
        return self.draft


class FakeParser:
    supported_extensions = frozenset({".pdf", ".docx"})

    def __init__(self, parsed: ParsedDocument) -> None:
        self.parsed = parsed

    async def parse(self, path: Path) -> ParsedDocument:
        return self.parsed


class FakeChunker:
    def __init__(self, size: int = 50) -> None:
        self.size = size

    def split(self, text: str) -> list[TextSpan]:
        return [
            TextSpan(text[i : i + self.size], i, min(i + self.size, len(text))) for i in range(0, len(text), self.size)
        ]


class FakeStorage:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def save(self, name: str, content: AsyncIterator[bytes], max_bytes: int) -> int:
        data = b""
        async for chunk in content:
            data += chunk
            if len(data) > max_bytes:
                msg = "too large"
                raise FileTooLargeError(msg)
        self.files[name] = data
        return len(data)

    async def local_path(self, name: str) -> Path:
        return Path("/fake") / name

    async def delete(self, name: str) -> None:
        self.files.pop(name, None)


class FakeStatuses:
    def __init__(self) -> None:
        self.tasks: dict[str, IndexingTask] = {}
        self.history: list[tuple[str, str]] = []

    async def save(self, task: IndexingTask) -> None:
        self.tasks[task.task_id] = task
        self.history.append((task.status.value, task.stage.value))

    async def get(self, task_id: str) -> IndexingTask | None:
        return self.tasks.get(task_id)


class FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    async def enqueue_indexing(self, task_id: str) -> None:
        self.enqueued.append(task_id)


async def bytes_stream(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


def utcnow() -> datetime:
    return datetime.now(UTC)
