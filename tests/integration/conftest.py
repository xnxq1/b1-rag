import os
from collections.abc import AsyncIterator, Iterator

import pytest
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

QDRANT_IMAGE = "qdrant/qdrant:v1.20.0"
REDIS_IMAGE = "redis:7.4-alpine"

os.environ.setdefault("TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE", "/var/run/docker.sock")


@pytest.fixture(scope="session")
def qdrant_url() -> Iterator[str]:
    from testcontainers.qdrant import QdrantContainer

    with QdrantContainer(QDRANT_IMAGE) as container:
        yield f"http://{container.rest_host_address}"


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    from testcontainers.redis import RedisContainer

    with RedisContainer(REDIS_IMAGE) as container:
        yield f"redis://{container.get_container_host_ip()}:{container.get_exposed_port(6379)}/0"


@pytest.fixture
async def qdrant_client(qdrant_url: str) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=qdrant_url)
    yield client
    await client.close()


@pytest.fixture
async def redis(redis_url: str) -> AsyncIterator[Redis]:
    client = Redis.from_url(redis_url)
    yield client
    await client.aclose()
