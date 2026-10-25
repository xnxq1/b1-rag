import asyncio

from rag.application.dto import AskResult, RankedFragment, SearchDiagnostics
from rag.application.ports import (
    AnswerGeneratorPort,
    EmbeddingsPort,
    HybridSearchResult,
    RerankerPort,
    SparseEncoderPort,
    VectorStorePort,
)
from rag.domain.entities import Answer, ScoredChunk
from rag.domain.exceptions import DocumentNotFoundError
from rag.domain.grounding import ground_answer, not_found_answer


class AskQuestion:
    def __init__(
        self,
        embeddings: EmbeddingsPort,
        sparse: SparseEncoderPort,
        vector_store: VectorStorePort,
        reranker: RerankerPort,
        generator: AnswerGeneratorPort,
        rerank_candidates: int,
        rerank_top_n: int,
        rerank_min_score: float,
    ) -> None:
        self._embeddings = embeddings
        self._sparse = sparse
        self._vector_store = vector_store
        self._reranker = reranker
        self._generator = generator
        self._rerank_candidates = rerank_candidates
        self._rerank_top_n = rerank_top_n
        self._rerank_min_score = rerank_min_score

    async def execute(self, doc_id: str, question: str, debug: bool = False) -> AskResult:
        if not await self._vector_store.document_exists(doc_id):
            msg = f"Документ {doc_id} не найден или ещё не проиндексирован"
            raise DocumentNotFoundError(msg)

        dense, sparse = await asyncio.gather(
            self._embeddings.embed_query(question),
            self._sparse.encode_query(question),
        )
        search = await self._vector_store.hybrid_search(
            doc_id,
            dense,
            sparse,
            limit=self._rerank_candidates,
            with_diagnostics=debug,
        )
        reranked: list[ScoredChunk] = []
        if search.candidates:
            reranked = await self._reranker.rerank(question, search.candidates, self._rerank_top_n)
        context = [c for c in reranked if c.score >= self._rerank_min_score]

        answer = not_found_answer()
        if context:
            draft = await self._generator.generate(question, context)
            answer = ground_answer(draft, context)

        diagnostics = None
        if debug:
            diagnostics = self._diagnostics(search, reranked, context, answer)
        return AskResult(answer=answer, diagnostics=diagnostics)

    @staticmethod
    def _diagnostics(
        search: HybridSearchResult,
        reranked: list[ScoredChunk],
        context: list[ScoredChunk],
        answer: Answer,
    ) -> SearchDiagnostics:
        in_context = {c.chunk.chunk_index for c in context}
        dense_ranks = {index: rank for rank, index in enumerate(search.dense_ranking or [], start=1)}
        sparse_ranks = {index: rank for rank, index in enumerate(search.sparse_ranking or [], start=1)}
        return SearchDiagnostics(
            dense_hits=len(dense_ranks),
            sparse_hits=len(sparse_ranks),
            fused_candidates=len(search.candidates),
            reranked=[
                RankedFragment(
                    chunk_index=c.chunk.chunk_index,
                    page_from=c.chunk.page_from,
                    page_to=c.chunk.page_to,
                    section=c.chunk.section,
                    rerank_score=round(c.score, 4),
                    dense_rank=dense_ranks.get(c.chunk.chunk_index),
                    sparse_rank=sparse_ranks.get(c.chunk.chunk_index),
                    used_in_context=c.chunk.chunk_index in in_context,
                )
                for c in reranked
            ],
            context_fragments=len(context),
            used_in_answer=sorted({c.chunk_index for c in answer.citations}),
            llm_called=bool(context),
        )
