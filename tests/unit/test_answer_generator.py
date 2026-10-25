import pytest
from langchain_core.exceptions import OutputParserException

from rag.adapters.outbound.llm.openrouter import LangChainAnswerGenerator, LLMAnswer, LLMCitation
from rag.domain.entities import ScoredChunk
from rag.domain.exceptions import InvalidModelResponseError
from tests.fakes import make_chunk


class StubChain:
    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.calls = 0

    async def ainvoke(self, _: dict) -> object:
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def generator_with(chain: StubChain) -> LangChainAnswerGenerator:
    generator = object.__new__(LangChainAnswerGenerator)
    generator._chain = chain
    return generator


CONTEXT = [ScoredChunk(make_chunk(0, "Неустойка 0,1% за каждый день."), 0.9)]


async def test_invalid_output_is_retried_once() -> None:
    valid = LLMAnswer(citations=[LLMCitation(fragment=1, quote="Неустойка 0,1%")], answer="0,1%", found=True)
    chain = StubChain(OutputParserException("bad json"), valid)

    draft = await generator_with(chain).generate("Какой штраф?", CONTEXT)

    assert chain.calls == 2
    assert draft.found
    assert draft.citations[0].quote == "Неустойка 0,1%"


async def test_repeated_invalid_output_raises_bad_gateway_error() -> None:
    chain = StubChain(OutputParserException("bad"), {"found": "maybe"})

    with pytest.raises(InvalidModelResponseError):
        await generator_with(chain).generate("Какой штраф?", CONTEXT)
    assert chain.calls == 2
