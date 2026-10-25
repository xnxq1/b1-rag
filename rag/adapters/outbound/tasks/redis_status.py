from pydantic import TypeAdapter
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

from rag.adapters.outbound.retry import RetryPolicy
from rag.domain.entities import IndexingTask

_SERVICE = "Хранилище статусов задач"
_KEY_PREFIX = "rag:task:"
_adapter = TypeAdapter(IndexingTask)


def is_transient(exc: BaseException) -> bool:
    return isinstance(exc, RedisConnectionError | RedisTimeoutError)


class RedisTaskStatusRepository:
    def __init__(self, redis: Redis, retry: RetryPolicy, ttl_seconds: int) -> None:
        self._redis = redis
        self._retry = retry
        self._ttl = ttl_seconds

    async def save(self, task: IndexingTask) -> None:
        payload = _adapter.dump_json(task)
        await self._retry.call(
            _SERVICE, lambda: self._redis.set(self._key(task.task_id), payload, ex=self._ttl), is_transient
        )

    async def get(self, task_id: str) -> IndexingTask | None:
        raw = await self._retry.call(_SERVICE, lambda: self._redis.get(self._key(task_id)), is_transient)
        return _adapter.validate_json(raw) if raw else None

    @staticmethod
    def _key(task_id: str) -> str:
        return f"{_KEY_PREFIX}{task_id}"
