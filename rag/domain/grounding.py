import re

from rag.domain.entities import Answer, AnswerDraft, Citation, ScoredChunk

NOT_FOUND_MESSAGE = "В документе нет сведений, необходимых для ответа на этот вопрос."

_ELLIPSIS = re.compile(r"\.{3,}|…")
_MIN_QUOTE_LEN = 8


def not_found_answer() -> Answer:
    return Answer(answer=NOT_FOUND_MESSAGE, found=False, citations=[])


def ground_answer(draft: AnswerDraft, context: list[ScoredChunk]) -> Answer:
    if not draft.found or not draft.answer.strip():
        return not_found_answer()

    citations: list[Citation] = []
    seen: set[tuple[int, int]] = set()
    for draft_citation in draft.citations:
        if not 1 <= draft_citation.fragment <= len(context):
            continue
        citation = _make_citation(context[draft_citation.fragment - 1], draft_citation.quote)
        key = (citation.chunk_index, citation.char_start)
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)

    if not citations:
        return not_found_answer()
    return Answer(answer=draft.answer.strip(), found=True, citations=citations)


def _make_citation(scored: ScoredChunk, quote: str) -> Citation:
    chunk = scored.chunk
    located = locate_quote(chunk.text, quote)
    if located is None:
        start, end = 0, len(chunk.text)
    else:
        start, end = located
    char_start, char_end = chunk.char_start + start, chunk.char_start + end
    return Citation(
        text=chunk.text[start:end],
        page=chunk.page_at(char_start),
        page_to=chunk.page_at(max(char_end - 1, char_start)),
        section=chunk.section_at(char_start),
        chunk_index=chunk.chunk_index,
        char_start=char_start,
        char_end=char_end,
    )


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    pieces = [p.strip() for p in _ELLIPSIS.split(quote) if p.strip()]
    if not pieces:
        return None
    piece = max(pieces, key=len)
    if len(piece) < _MIN_QUOTE_LEN:
        return None
    pattern = r"\s+".join(re.escape(word) for word in piece.split())
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end()
