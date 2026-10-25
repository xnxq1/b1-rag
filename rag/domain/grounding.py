import re

from rag.domain.entities import Answer, AnswerDraft, Citation, ScoredChunk

NOT_FOUND_MESSAGE = "В документе нет сведений, необходимых для ответа на этот вопрос."

_ELLIPSIS = re.compile(r"\.{3,}|…")
_MIN_QUOTE_LEN = 8
_MIN_QUOTE_WORDS = 2


def not_found_answer() -> Answer:
    return Answer(answer=NOT_FOUND_MESSAGE, found=False, citations=[])


def ground_answer(draft: AnswerDraft, context: list[ScoredChunk]) -> Answer:
    if not draft.found or not draft.answer.strip():
        return not_found_answer()

    citations: list[Citation] = []
    for draft_citation in draft.citations:
        if not 1 <= draft_citation.fragment <= len(context):
            continue
        citation = _make_citation(context[draft_citation.fragment - 1], draft_citation.quote)
        if citation is None or any(_overlaps(citation, existing) for existing in citations):
            continue
        citations.append(citation)

    if not citations:
        return not_found_answer()
    return Answer(answer=draft.answer.strip(), found=True, citations=citations)


def _overlaps(a: Citation, b: Citation) -> bool:
    return a.char_start < b.char_end and b.char_start < a.char_end


def _make_citation(scored: ScoredChunk, quote: str) -> Citation | None:
    chunk = scored.chunk
    located = locate_quote(chunk.text, quote)
    if located is None:
        return None
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
    words = max(pieces, key=len).split()
    for size in range(len(words), _MIN_QUOTE_WORDS - 1, -1):
        for start in range(len(words) - size + 1):
            window = words[start : start + size]
            if len(" ".join(window)) < _MIN_QUOTE_LEN:
                continue
            pattern = r"\s+".join(re.escape(word) for word in window)
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match is not None:
                return match.start(), match.end()
    return None
