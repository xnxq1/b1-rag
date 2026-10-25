from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.domain.entities import TextSpan

_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


class RecursiveChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_SEPARATORS,
            keep_separator="end",
            add_start_index=True,
        )

    def split(self, text: str) -> list[TextSpan]:
        spans = []
        for document in self._splitter.create_documents([text]):
            start = document.metadata["start_index"]
            spans.append(
                TextSpan(text=document.page_content, char_start=start, char_end=start + len(document.page_content))
            )
        return spans
