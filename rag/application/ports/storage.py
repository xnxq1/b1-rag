from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol


class FileStoragePort(Protocol):
    async def save(self, name: str, content: AsyncIterator[bytes], max_bytes: int) -> int: ...

    async def local_path(self, name: str) -> Path: ...

    async def delete(self, name: str) -> None: ...
