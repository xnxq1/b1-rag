from bisect import bisect_right

from rag.domain.entities import Chunk, Heading, PageSpan, ParsedDocument, TextSpan


class _PageLookup:
    def __init__(self, pages: list[PageSpan]) -> None:
        self._pages = sorted(pages, key=lambda p: p.char_start)
        self._starts = [p.char_start for p in self._pages]

    def page_at(self, offset: int) -> int | None:
        if not self._pages:
            return None
        idx = bisect_right(self._starts, offset) - 1
        return self._pages[max(idx, 0)].page


class _SectionLookup:
    def __init__(self, headings: list[Heading]) -> None:
        self._headings = sorted(headings, key=lambda h: h.char_start)
        self._starts = [h.char_start for h in self._headings]

    def section_at(self, offset: int) -> str | None:
        idx = bisect_right(self._starts, offset) - 1
        return self._headings[idx].title if idx >= 0 else None


def build_chunks(doc_id: str, doc_title: str, parsed: ParsedDocument, spans: list[TextSpan]) -> list[Chunk]:
    pages = _PageLookup(parsed.pages)
    sections = _SectionLookup(parsed.headings)
    chunks = []
    for index, span in enumerate(spans):
        last_char = max(span.char_end - 1, span.char_start)
        inside = range(span.char_start + 1, span.char_end)
        chunks.append(
            Chunk(
                doc_id=doc_id,
                doc_title=doc_title,
                chunk_index=index,
                text=span.text,
                char_start=span.char_start,
                char_end=span.char_end,
                page_from=pages.page_at(span.char_start),
                page_to=pages.page_at(last_char),
                section=sections.section_at(span.char_start),
                page_marks=[p for p in parsed.pages if p.char_start in inside],
                heading_marks=[h for h in parsed.headings if h.char_start in inside],
            ),
        )
    return chunks
