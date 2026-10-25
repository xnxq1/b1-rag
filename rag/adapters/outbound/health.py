from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis


class QdrantCheck:
    name = "qdrant"

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def is_available(self) -> bool:
        await self._client.info()
        return True


class RedisCheck:
    name = "redis"

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def is_available(self) -> bool:
        return bool(await self._redis.ping())
