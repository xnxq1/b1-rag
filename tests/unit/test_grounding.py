from rag.domain.entities import AnswerDraft, DraftCitation, Heading, PageSpan, ScoredChunk
from rag.domain.grounding import NOT_FOUND_MESSAGE, ground_answer, locate_quote
from tests.fakes import make_chunk

PENALTY = "4.2. За нарушение сроков поставки Поставщик уплачивает Покупателю неустойку в размере 0,1% за каждый день."


def context() -> list[ScoredChunk]:
    return [
        ScoredChunk(make_chunk(3, PENALTY, page=12, section="4.2 Штрафные санкции"), 0.9),
        ScoredChunk(make_chunk(7, "Покупатель оплачивает товар в течение 10 дней.", page=5), 0.5),
    ]


def test_quote_is_located_and_offsets_point_into_document() -> None:
    draft = AnswerDraft(True, "Штраф 0,1% в день.", [DraftCitation(1, "неустойку  в размере 0,1%")])
    answer = ground_answer(draft, context())

    assert answer.found
    citation = answer.citations[0]
    assert citation.text == "неустойку в размере 0,1%"
    assert citation.page == 12
    assert citation.section == "4.2 Штрафные санкции"
    chunk_start = 3 * 1000
    assert PENALTY[citation.char_start - chunk_start : citation.char_end - chunk_start] == citation.text


def test_unverified_quote_is_dropped_and_answer_becomes_not_found() -> None:
    draft = AnswerDraft(True, "Штраф 0,1%.", [DraftCitation(1, "штраф составляет одну десятую процента")])
    answer = ground_answer(draft, context())

    assert not answer.found
    assert answer.citations == []


def test_only_verified_quotes_remain() -> None:
    draft = AnswerDraft(
        True,
        "Штраф 0,1%.",
        [DraftCitation(1, "штраф составляет одну десятую процента"), DraftCitation(1, "неустойку в размере 0,1%")],
    )
    answer = ground_answer(draft, context())

    assert [c.text for c in answer.citations] == ["неустойку в размере 0,1%"]


def test_citation_outside_context_is_dropped_and_answer_becomes_not_found() -> None:
    draft = AnswerDraft(True, "Ответ из общих знаний", [DraftCitation(5, "что-то")])
    answer = ground_answer(draft, context())

    assert not answer.found
    assert answer.citations == []
    assert answer.answer == NOT_FOUND_MESSAGE


def test_found_without_citations_is_not_found() -> None:
    answer = ground_answer(AnswerDraft(True, "Ответ", []), context())
    assert not answer.found


def test_model_not_found_is_respected() -> None:
    answer = ground_answer(AnswerDraft(False, "", [DraftCitation(1, "неустойку")]), context())
    assert not answer.found
    assert answer.citations == []


def test_duplicate_citations_are_merged() -> None:
    quote = "Поставщик уплачивает Покупателю"
    answer = ground_answer(AnswerDraft(True, "Да", [DraftCitation(1, quote), DraftCitation(1, quote)]), context())
    assert len(answer.citations) == 1


def test_locate_quote_with_ellipsis_uses_longest_piece() -> None:
    located = locate_quote(PENALTY, "За нарушение … уплачивает Покупателю неустойку")
    assert located is not None
    assert PENALTY[located[0] : located[1]] == "уплачивает Покупателю неустойку"


def test_citation_page_and_section_follow_quote_position_inside_chunk() -> None:
    text = "3.4. Досрочная поставка по согласию.\n4.2 Штрафные санкции\n4.2.1. Неустойка 0,1% за день."
    chunk = make_chunk(0, text, page=2, section="3. СРОКИ ПОСТАВКИ")
    heading_at = text.index("4.2 Штрафные")
    chunk.heading_marks = [Heading(heading_at, "4.2 Штрафные санкции")]
    chunk.page_marks = [PageSpan(3, heading_at, len(text))]
    chunk.page_to = 3

    answer = ground_answer(
        AnswerDraft(True, "0,1%", [DraftCitation(1, "Неустойка 0,1% за день")]), [ScoredChunk(chunk, 1)]
    )

    citation = answer.citations[0]
    assert (citation.page, citation.page_to, citation.section) == (3, 3, "4.2 Штрафные санкции")


def test_quote_with_changed_word_form_is_located_by_longest_word_run() -> None:
    text = "1.1. Поставщик обязуется передать\nв собственность Покупателя, АО «Северная верфь», серверное оборудование."
    located = locate_quote(text, "Покупатель, АО «Северная верфь»")
    assert located is not None
    assert text[located[0] : located[1]] == "АО «Северная верфь»"


def test_single_word_overlap_is_not_a_quote() -> None:
    assert locate_quote(PENALTY, "Неустойка определяется законом") is None


def test_overlapping_citations_of_same_place_are_merged() -> None:
    draft = AnswerDraft(
        True,
        "0,1%",
        [DraftCitation(1, "Поставщик уплачивает Покупателю неустойку"), DraftCitation(1, "уплачивает Покупателю")],
    )
    answer = ground_answer(draft, context())
    assert [c.text for c in answer.citations] == ["Поставщик уплачивает Покупателю неустойку"]
