import pytest

from rag.adapters.outbound.chunking.recursive import RecursiveChunker
from rag.adapters.outbound.parsing.headings import detect_headings, is_heading
from rag.adapters.outbound.sparse.bm25 import add_exact_tokens


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("пункт 4.2", ["4_2"]),
        ("неустойка 0,1%", ["0_1"]),
        ("до 12.03.2024", ["12_03_2024"]),
        ("сумма 1 500 000 руб.", ["1500000"]),
        ("ГОСТ 12.1.004-91", ["12_1_004_91"]),
    ],
)
def test_exact_tokens_are_appended(text: str, expected: list[str]) -> None:
    assert add_exact_tokens(text).split("\n")[1].split() == expected


def test_text_without_identifiers_is_unchanged() -> None:
    assert add_exact_tokens("просто текст") == "просто текст"


@pytest.mark.parametrize(
    "line",
    ["4.2 Штрафные санкции", "4. ОТВЕТСТВЕННОСТЬ СТОРОН", "Статья 5. Форс-мажор", "ПРЕДМЕТ ДОГОВОРА", "Приложение № 1"],
)
def test_headings_detected(line: str) -> None:
    assert is_heading(line)


@pytest.mark.parametrize(
    "line",
    [
        "4.2.1. Поставщик обязан поставить товар в срок.",
        "Покупатель вправе отказаться от товара;",
        "обычная строка текста без признаков заголовка",
        "ООО",
    ],
)
def test_clauses_are_not_headings(line: str) -> None:
    assert not is_heading(line)


def test_wrapped_clause_line_is_not_heading() -> None:
    text = (
        "1.1. Поставщик, ООО «ТехноСнаб», обязуется передать\nв собственность Покупателя товар.\n"
        "1.2. Наименование, количество и комплектность Товара определяются\nСпецификацией.\n"
        "4.2 Штрафные санкции\n4.2.1. Текст."
    )
    assert [h.title for h in detect_headings(text)] == ["4.2 Штрафные санкции"]


def test_detect_headings_offsets() -> None:
    text = "Вступление\n4.2 Штрафные санкции\nТекст пункта."
    headings = detect_headings(text)
    assert [(h.title, text[h.char_start : h.char_start + 3]) for h in headings] == [("4.2 Штрафные санкции", "4.2")]


def test_chunker_offsets_match_source_text() -> None:
    text = "\n\n".join(f"{i}. Пункт номер {i}. " + "слово " * 40 for i in range(1, 20))
    spans = RecursiveChunker(chunk_size=300, chunk_overlap=50).split(text)

    assert len(spans) > 1
    for span in spans:
        assert text[span.char_start : span.char_end] == span.text
        assert len(span.text) <= 300
