from celery import Celery

from rag.config import Settings

INDEX_DOCUMENT_TASK = "rag.index_document"
INDEXING_QUEUE = "indexing"


def build_celery(settings: Settings) -> Celery:
    app = Celery("rag", broker=settings.redis_url)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        task_default_queue=INDEXING_QUEUE,
        task_ignore_result=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        broker_transport_options={"visibility_timeout": 3600},
        broker_connection_retry_on_startup=True,
    )
    return app
