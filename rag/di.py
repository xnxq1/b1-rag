from collections.abc import AsyncIterator

import httpx
from celery import Celery
from dishka import Provider, Scope, from_context, make_async_container, provide
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from rag.adapters.inbound.api.builder import AppBuilder
from rag.adapters.inbound.api.routers.documents import DocumentsRouter
from rag.adapters.inbound.api.routers.health import HealthRouter
from rag.adapters.inbound.api.routers.tasks import TasksRouter
from rag.adapters.outbound.chunking.recursive import RecursiveChunker
from rag.adapters.outbound.embeddings.openrouter import OpenRouterEmbeddings
from rag.adapters.outbound.llm.openrouter import LangChainAnswerGenerator
from rag.adapters.outbound.parsing.dispatcher import FileDocumentParser
from rag.adapters.outbound.rerank.openrouter import OpenRouterReranker
from rag.adapters.outbound.retry import RetryPolicy
from rag.adapters.outbound.sparse.bm25 import FastEmbedBm25Encoder
from rag.adapters.outbound.storage.local import LocalFileStorage
from rag.adapters.outbound.tasks.celery_app import build_celery
from rag.adapters.outbound.tasks.celery_queue import CeleryTaskQueue
from rag.adapters.outbound.tasks.redis_status import RedisTaskStatusRepository
from rag.adapters.outbound.vector.qdrant_store import QdrantVectorStore
from rag.application.ports import (
    AnswerGeneratorPort,
    ChunkerPort,
    DocumentParserPort,
    EmbeddingsPort,
    FileStoragePort,
    RerankerPort,
    SparseEncoderPort,
    TaskQueuePort,
    TaskStatusRepository,
    VectorStorePort,
)
from rag.application.use_cases import AskQuestion, GetIndexingStatus, IndexDocument, ListDocuments, UploadDocument
from rag.config import Settings, load_settings
from rag.logging_setup import setup_logging


def _openrouter_headers(settings: Settings) -> dict[str, str]:
    return {"X-Title": settings.openrouter_app_title}


class AppProvider(Provider):
    scope = Scope.APP

    settings = from_context(provides=Settings, scope=Scope.APP)
    celery = from_context(provides=Celery, scope=Scope.APP)

    @provide
    def retry(self, settings: Settings) -> RetryPolicy:
        return RetryPolicy(attempts=settings.retry_attempts, max_wait_seconds=settings.retry_max_wait_seconds)

    @provide
    async def qdrant_client(self, settings: Settings) -> AsyncIterator[AsyncQdrantClient]:
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            timeout=settings.qdrant_timeout_seconds,
        )
        try:
            yield client
        finally:
            await client.close()

    @provide
    async def redis(self, settings: Settings) -> AsyncIterator[Redis]:
        client = Redis.from_url(settings.redis_url)
        try:
            yield client
        finally:
            await client.aclose()

    @provide
    async def http_client(self, settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
        client = httpx.AsyncClient(
            base_url=f"{settings.openrouter_base_url.rstrip('/')}/",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
                **_openrouter_headers(settings),
            },
            timeout=settings.rerank_timeout_seconds,
        )
        try:
            yield client
        finally:
            await client.aclose()

    @provide
    def embeddings(self, settings: Settings) -> EmbeddingsPort:
        model = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            default_headers=_openrouter_headers(settings),
            check_embedding_ctx_length=False,
            chunk_size=settings.embedding_batch_size,
            max_retries=settings.llm_max_retries,
            request_timeout=settings.llm_timeout_seconds,
            model_kwargs={"encoding_format": "float"},
        )
        return OpenRouterEmbeddings(model, dimension=settings.embedding_dim)

    @provide
    def sparse(self, settings: Settings) -> SparseEncoderPort:
        return FastEmbedBm25Encoder(language=settings.bm25_language, avg_len=settings.bm25_avg_len)

    @provide
    def vector_store(self, client: AsyncQdrantClient, retry: RetryPolicy, settings: Settings) -> VectorStorePort:
        return QdrantVectorStore(
            client=client,
            retry=retry,
            collection=settings.qdrant_collection,
            dense_dim=settings.embedding_dim,
            fusion=settings.search_fusion,
            dense_limit=settings.search_dense_limit,
            sparse_limit=settings.search_sparse_limit,
            upsert_batch_size=settings.qdrant_upsert_batch_size,
        )

    @provide
    def reranker(self, client: httpx.AsyncClient, retry: RetryPolicy, settings: Settings) -> RerankerPort:
        return OpenRouterReranker(client=client, retry=retry, model=settings.rerank_model)

    @provide
    def generator(self, settings: Settings) -> AnswerGeneratorPort:
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers=_openrouter_headers(settings),
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        return LangChainAnswerGenerator(llm, structured_method=settings.llm_structured_method)

    @provide
    def parser(self) -> DocumentParserPort:
        return FileDocumentParser()

    @provide
    def chunker(self, settings: Settings) -> ChunkerPort:
        return RecursiveChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    @provide
    def storage(self, settings: Settings) -> FileStoragePort:
        return LocalFileStorage(settings.storage_dir)

    @provide
    def statuses(self, redis: Redis, retry: RetryPolicy, settings: Settings) -> TaskStatusRepository:
        return RedisTaskStatusRepository(redis, retry, ttl_seconds=settings.task_status_ttl_seconds)

    @provide
    def queue(self, celery: Celery) -> TaskQueuePort:
        return CeleryTaskQueue(celery)

    @provide
    def upload_document(
        self,
        storage: FileStoragePort,
        statuses: TaskStatusRepository,
        queue: TaskQueuePort,
        parser: DocumentParserPort,
        settings: Settings,
    ) -> UploadDocument:
        return UploadDocument(
            storage=storage,
            statuses=statuses,
            queue=queue,
            supported_extensions=parser.supported_extensions,
            max_bytes=settings.max_upload_bytes,
        )

    @provide
    def index_document(
        self,
        statuses: TaskStatusRepository,
        storage: FileStoragePort,
        parser: DocumentParserPort,
        chunker: ChunkerPort,
        embeddings: EmbeddingsPort,
        sparse: SparseEncoderPort,
        vector_store: VectorStorePort,
        settings: Settings,
    ) -> IndexDocument:
        return IndexDocument(
            statuses=statuses,
            storage=storage,
            parser=parser,
            chunker=chunker,
            embeddings=embeddings,
            sparse=sparse,
            vector_store=vector_store,
            embedding_batch_size=settings.embedding_batch_size,
        )

    @provide
    def get_indexing_status(self, statuses: TaskStatusRepository) -> GetIndexingStatus:
        return GetIndexingStatus(statuses)

    @provide
    def list_documents(self, vector_store: VectorStorePort) -> ListDocuments:
        return ListDocuments(vector_store)

    @provide
    def ask_question(
        self,
        embeddings: EmbeddingsPort,
        sparse: SparseEncoderPort,
        vector_store: VectorStorePort,
        reranker: RerankerPort,
        generator: AnswerGeneratorPort,
        settings: Settings,
    ) -> AskQuestion:
        return AskQuestion(
            embeddings=embeddings,
            sparse=sparse,
            vector_store=vector_store,
            reranker=reranker,
            generator=generator,
            rerank_candidates=settings.rerank_candidates,
            rerank_top_n=settings.rerank_top_n,
            rerank_min_score=settings.rerank_min_score,
        )


class ApiProvider(Provider):
    scope = Scope.APP

    @provide
    def documents_router(
        self,
        upload_document: UploadDocument,
        list_documents: ListDocuments,
        ask_question: AskQuestion,
    ) -> DocumentsRouter:
        return DocumentsRouter(
            upload_document=upload_document,
            list_documents=list_documents,
            ask_question=ask_question,
        )

    @provide
    def tasks_router(self, get_indexing_status: GetIndexingStatus) -> TasksRouter:
        return TasksRouter(get_indexing_status=get_indexing_status)

    @provide
    def health_router(self) -> HealthRouter:
        return HealthRouter()

    @provide
    def app_builder(
        self,
        documents_router: DocumentsRouter,
        tasks_router: TasksRouter,
        health_router: HealthRouter,
        vector_store: VectorStorePort,
        settings: Settings,
    ) -> AppBuilder:
        return AppBuilder(
            routers=[documents_router, tasks_router, health_router],
            vector_store=vector_store,
            max_upload_bytes=settings.max_upload_bytes,
        )


def build_providers() -> tuple[Provider, ...]:
    return (AppProvider(), ApiProvider())


settings = load_settings()
setup_logging(settings.log_level)
celery_app = build_celery(settings)
container = make_async_container(*build_providers(), context={Settings: settings, Celery: celery_app})
