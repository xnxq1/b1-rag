from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class SparseEncoderPort(Protocol):
    async def encode_documents(self, texts: list[str]) -> list[SparseVector]: ...

    async def encode_query(self, text: str) -> SparseVector: ...
