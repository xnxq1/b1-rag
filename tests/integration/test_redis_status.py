from datetime import UTC, datetime

import pytest
from redis.asyncio import Redis

from rag.adapters.outbound.retry import RetryPolicy
from rag.adapters.outbound.tasks.redis_status import RedisTaskStatusRepository
from rag.domain.entities import IndexingStage, IndexingTask, TaskStatus

pytestmark = pytest.mark.integration


async def test_status_roundtrip(redis: Redis) -> None:
    repo = RedisTaskStatusRepository(redis, RetryPolicy(attempts=2, max_wait_seconds=0.1), ttl_seconds=60)
    now = datetime.now(UTC)
    task = IndexingTask(
        "t1", "t1", "Договор", "Договор.pdf", TaskStatus.RUNNING, IndexingStage.EMBEDDING, 0.5, now, now
    )

    await repo.save(task)

    assert await repo.get("t1") == task
    assert await repo.get("missing") is None
    assert 0 < await redis.ttl("rag:task:t1") <= 60
