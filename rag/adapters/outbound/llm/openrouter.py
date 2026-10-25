import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from rag.adapters.outbound.llm.prompts import HUMAN_PROMPT, SYSTEM_PROMPT, format_fragment
from rag.adapters.outbound.openai_errors import translate_openai_errors
from rag.domain.entities import AnswerDraft, DraftCitation, ScoredChunk
from rag.domain.exceptions import InvalidModelResponseError

logger = logging.getLogger(__name__)

_PARSE_ATTEMPTS = 2


class LLMCitation(BaseModel):
    fragment: int = Field(description="Номер фрагмента из контекста, например 2 для [2]")
    quote: str = Field(description="Дословная цитата из этого фрагмента, подтверждающая ответ")


class LLMAnswer(BaseModel):
    citations: list[LLMCitation] = Field(description="Фрагменты, на которые опирается ответ; пусто, если ответа нет")
    answer: str = Field(description="Ответ на вопрос; пустая строка, если во фрагментах нет сведений")
    found: bool = Field(description="true, только если ответ полностью подтверждается фрагментами")


class LangChainAnswerGenerator:
    def __init__(self, llm: BaseChatModel, structured_method: str) -> None:
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
        self._chain = prompt | llm.with_structured_output(LLMAnswer, method=structured_method)

    async def generate(self, question: str, context: list[ScoredChunk]) -> AnswerDraft:
        fragments = "\n\n".join(
            format_fragment(i, c.chunk.text, c.chunk.page_from, c.chunk.page_to, c.chunk.section)
            for i, c in enumerate(context, start=1)
        )
        variables = {"doc_title": context[0].chunk.doc_title, "fragments": fragments, "question": question}
        for attempt in range(1, _PARSE_ATTEMPTS + 1):
            try:
                with translate_openai_errors("Языковая модель"):
                    result = await self._chain.ainvoke(variables)
                answer = result if isinstance(result, LLMAnswer) else LLMAnswer.model_validate(result)
            except (OutputParserException, ValidationError) as exc:
                if attempt == _PARSE_ATTEMPTS:
                    msg = "Языковая модель вернула ответ в неверном формате"
                    raise InvalidModelResponseError(msg) from exc
                logger.warning("Ответ модели не прошёл валидацию, повтор: %s", exc)
                continue
            return AnswerDraft(
                found=answer.found,
                answer=answer.answer,
                citations=[DraftCitation(fragment=c.fragment, quote=c.quote) for c in answer.citations],
            )
        msg = "Языковая модель не вернула ответ"
        raise InvalidModelResponseError(msg)
