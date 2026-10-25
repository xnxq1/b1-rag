from http import HTTPStatus

import httpx

from rag.adapters.outbound.retry import RetryPolicy
from rag.domain.entities import ScoredChunk

_SERVICE = "Сервис переранжирования"


class RerankResponseError(RuntimeError):
    pass


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= HTTPStatus.INTERNAL_SERVER_ERROR or status == HTTPStatus.TOO_MANY_REQUESTS
    return False


class OpenRouterReranker:
    def __init__(self, client: httpx.AsyncClient, retry: RetryPolicy, model: str) -> None:
        self._client = client
        self._retry = retry
        self._model = model

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        body = {
            "model": self._model,
            "query": query,
            "documents": [c.chunk.text for c in candidates],
            "top_n": min(top_n, len(candidates)),
        }
        data = await self._retry.call(_SERVICE, lambda: self._post(body), is_transient)
        try:
            results = sorted(data["results"], key=lambda r: r["relevance_score"], reverse=True)
            return [
                ScoredChunk(chunk=candidates[r["index"]].chunk, score=float(r["relevance_score"]))
                for r in results[:top_n]
            ]
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"Неожиданный ответ реранкера: {str(data)[:300]}"
            raise RerankResponseError(msg) from exc

    async def _post(self, body: dict) -> dict:
        response = await self._client.post("rerank", json=body)
        response.raise_for_status()
        return response.json()
