import asyncio
from collections.abc import Callable
from pathlib import Path

from rag.adapters.outbound.parsing.docx import parse_docx
from rag.adapters.outbound.parsing.pdf import parse_pdf
from rag.domain.entities import ParsedDocument
from rag.domain.exceptions import UnsupportedFileTypeError

_PARSERS: dict[str, Callable[[Path], ParsedDocument]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}


class FileDocumentParser:
    supported_extensions = frozenset(_PARSERS)

    async def parse(self, path: Path) -> ParsedDocument:
        parser = _PARSERS.get(path.suffix.lower())
        if parser is None:
            msg = f"Неподдерживаемый формат файла «{path.suffix}»"
            raise UnsupportedFileTypeError(msg)
        return await asyncio.to_thread(parser, path)
