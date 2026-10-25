from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from rag.adapters.outbound.llm.prompts import HUMAN_PROMPT, SYSTEM_PROMPT, format_fragment
from rag.adapters.outbound.openai_errors import translate_openai_errors
from rag.domain.entities import AnswerDraft, DraftCitation, ScoredChunk
from rag.domain.exceptions import ExternalServiceUnavailableError


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
        try:
            with translate_openai_errors("Языковая модель"):
                result = await self._chain.ainvoke(variables)
        except (OutputParserException, ValidationError) as exc:
            msg = "Языковая модель вернула ответ в неверном формате, попробуйте ещё раз"
            raise ExternalServiceUnavailableError(msg) from exc

        if not isinstance(result, LLMAnswer):
            result = LLMAnswer.model_validate(result)
        return AnswerDraft(
            found=result.found,
            answer=result.answer,
            citations=[DraftCitation(fragment=c.fragment, quote=c.quote) for c in result.citations],
        )
