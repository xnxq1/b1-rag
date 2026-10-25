import re
from pathlib import Path

import pymupdf

from rag.adapters.outbound.parsing.headings import detect_headings
from rag.domain.entities import PageSpan, ParsedDocument

PAGE_SEPARATOR = "\n\n"
_SPACES = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    lines = (_SPACES.sub(" ", line).strip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def parse_pdf(path: Path) -> ParsedDocument:
    parts: list[str] = []
    pages: list[PageSpan] = []
    offset = 0
    with pymupdf.open(path) as pdf:
        pages_total = pdf.page_count
        for number, page in enumerate(pdf, start=1):
            text = _clean(page.get_text("text", sort=True))
            if parts:
                parts.append(PAGE_SEPARATOR)
                offset += len(PAGE_SEPARATOR)
            pages.append(PageSpan(page=number, char_start=offset, char_end=offset + len(text)))
            parts.append(text)
            offset += len(text)
    full_text = "".join(parts)
    return ParsedDocument(text=full_text, pages=pages, headings=detect_headings(full_text), pages_total=pages_total)
