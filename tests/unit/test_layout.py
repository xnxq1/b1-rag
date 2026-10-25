from rag.domain.entities import Heading, PageSpan, ParsedDocument, TextSpan
from rag.domain.layout import build_chunks


def test_pages_and_sections_are_resolved_by_offsets() -> None:
    parsed = ParsedDocument(
        text="x" * 300,
        pages=[PageSpan(1, 0, 100), PageSpan(2, 102, 200), PageSpan(3, 202, 300)],
        headings=[Heading(0, "1. Предмет"), Heading(150, "2. Цена")],
    )
    spans = [TextSpan("a", 10, 90), TextSpan("b", 90, 160), TextSpan("c", 160, 260)]

    chunks = build_chunks("doc", "Договор", parsed, spans)

    assert [(c.page_from, c.page_to) for c in chunks] == [(1, 1), (1, 2), (2, 3)]
    assert [c.section for c in chunks] == ["1. Предмет", "1. Предмет", "2. Цена"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert [m.page for m in chunks[1].page_marks] == [2]
    assert [h.title for h in chunks[1].heading_marks] == ["2. Цена"]
    assert chunks[1].section_at(155) == "2. Цена"
    assert chunks[1].page_at(95) == 1


def test_without_page_markup_pages_are_none() -> None:
    parsed = ParsedDocument(text="текст", pages=[], headings=[])
    chunk = build_chunks("doc", "Договор", parsed, [TextSpan("текст", 0, 5)])[0]
    assert chunk.page_from is None
    assert chunk.section is None
