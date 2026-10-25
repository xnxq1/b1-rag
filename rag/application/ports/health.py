from typing import Protocol


class DependencyCheckPort(Protocol):
    name: str

    async def is_available(self) -> bool: ...
