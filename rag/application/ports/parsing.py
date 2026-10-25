from pathlib import Path
from typing import Protocol

from rag.domain.entities import ParsedDocument, TextSpan


class DocumentParserPort(Protocol):
    supported_extensions: frozenset[str]

    async def parse(self, path: Path) -> ParsedDocument: ...


class ChunkerPort(Protocol):
    def split(self, text: str) -> list[TextSpan]: ...
