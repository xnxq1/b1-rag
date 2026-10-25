from collections.abc import Iterator
from contextlib import contextmanager

import openai

from rag.domain.exceptions import ExternalServiceUnavailableError

_TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)


@contextmanager
def translate_openai_errors(service: str) -> Iterator[None]:
    try:
        yield
    except _TRANSIENT as exc:
        msg = f"{service} временно недоступен, попробуйте позже"
        raise ExternalServiceUnavailableError(msg) from exc
