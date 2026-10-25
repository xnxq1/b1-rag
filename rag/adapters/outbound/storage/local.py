from collections.abc import AsyncIterator
from pathlib import Path

import anyio

from rag.domain.exceptions import FileTooLargeError


class LocalFileStorage:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, name: str, content: AsyncIterator[bytes], max_bytes: int) -> int:
        path = self._path(name)
        size = 0
        async with await anyio.open_file(path, "wb") as file:
            async for chunk in content:
                size += len(chunk)
                if size > max_bytes:
                    break
                await file.write(chunk)
        if size > max_bytes:
            await self.delete(name)
            raise FileTooLargeError.for_limit(max_bytes)
        return size

    async def local_path(self, name: str) -> Path:
        path = self._path(name)
        if not path.exists():
            msg = f"Файл {name} не найден в хранилище"
            raise FileNotFoundError(msg)
        return path

    async def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def _path(self, name: str) -> Path:
        path = (self._base_dir / name).resolve()
        if path.parent != self._base_dir.resolve():
            msg = f"Недопустимое имя файла: {name}"
            raise ValueError(msg)
        return path
