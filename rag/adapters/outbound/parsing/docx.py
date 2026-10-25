from collections.abc import Iterator
from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from rag.adapters.outbound.parsing.headings import is_heading, normalize_heading
from rag.domain.entities import Heading, PageSpan, ParsedDocument

BLOCK_SEPARATOR = "\n"
_HEADING_STYLES = ("heading", "заголовок", "title", "название")


def _blocks(document: DocxDocument) -> Iterator[Paragraph | Table]:
    for element in document.element.body.iterchildren():
        if element.tag == qn("w:p"):
            yield Paragraph(element, document)
        elif element.tag == qn("w:tbl"):
            yield Table(element, document)


def _rendered_breaks(paragraph: Paragraph) -> int:
    return len(paragraph._p.xpath(".//w:lastRenderedPageBreak"))  # noqa: SLF001


def _explicit_breaks(paragraph: Paragraph) -> int:
    count = len(paragraph._p.xpath('.//w:br[@w:type="page"]'))  # noqa: SLF001
    if paragraph.paragraph_format.page_break_before:
        count += 1
    return count


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = " ".join(cell.text.split())
            if text and (not cells or cells[-1] != text):
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _is_styled_heading(paragraph: Paragraph) -> bool:
    style = (paragraph.style.name if paragraph.style is not None else "") or ""
    return style.lower().startswith(_HEADING_STYLES)


def parse_docx(path: Path) -> ParsedDocument:
    document = docx.Document(str(path))
    blocks = list(_blocks(document))
    paragraphs = [b for b in blocks if isinstance(b, Paragraph)]
    use_rendered = any(_rendered_breaks(p) for p in paragraphs)
    has_page_info = use_rendered or any(_explicit_breaks(p) for p in paragraphs)

    parts: list[str] = []
    headings: list[Heading] = []
    pages: list[PageSpan] = []
    page, page_start, offset = 1, 0, 0

    for block in blocks:
        if isinstance(block, Paragraph):
            breaks = _rendered_breaks(block) if use_rendered else _explicit_breaks(block)
            text = block.text.strip()
        else:
            breaks = 0
            text = _table_text(block)

        if breaks and has_page_info and offset > page_start:
            pages.append(PageSpan(page=page, char_start=page_start, char_end=offset))
            page += breaks
            page_start = offset
        elif breaks:
            page += breaks

        if not text:
            continue
        if parts:
            parts.append(BLOCK_SEPARATOR)
            offset += len(BLOCK_SEPARATOR)
        if isinstance(block, Paragraph) and (_is_styled_heading(block) or is_heading(text)):
            headings.append(Heading(char_start=offset, title=normalize_heading(text)))
        parts.append(text)
        offset += len(text)

    if has_page_info:
        pages.append(PageSpan(page=page, char_start=page_start, char_end=offset))
    return ParsedDocument(
        text="".join(parts),
        pages=pages,
        headings=headings,
        pages_total=page if has_page_info else None,
    )
