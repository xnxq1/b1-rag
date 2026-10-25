from dataclasses import dataclass

from rag.domain.entities import Answer


@dataclass(frozen=True, slots=True)
class RankedFragment:
    chunk_index: int
    page_from: int | None
    page_to: int | None
    section: str | None
    rerank_score: float
    used_in_context: bool


@dataclass(slots=True)
class SearchDiagnostics:
    dense_hits: int
    sparse_hits: int
    fused_candidates: int
    reranked: list[RankedFragment]
    context_fragments: int
    used_in_answer: list[int]
    llm_called: bool


@dataclass(slots=True)
class AskResult:
    answer: Answer
    diagnostics: SearchDiagnostics | None = None
