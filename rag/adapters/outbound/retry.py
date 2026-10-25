import logging
from collections.abc import Awaitable, Callable

from tenacity import AsyncRetrying, RetryError, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from rag.domain.exceptions import ExternalServiceUnavailableError

logger = logging.getLogger(__name__)


class RetryPolicy:
    def __init__(self, attempts: int, max_wait_seconds: float) -> None:
        self._attempts = attempts
        self._max_wait = max_wait_seconds

    async def call[T](
        self,
        service: str,
        fn: Callable[[], Awaitable[T]],
        is_transient: Callable[[BaseException], bool],
    ) -> T:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._attempts),
            wait=wait_exponential_jitter(initial=0.5, max=self._max_wait),
            retry=retry_if_exception(is_transient),
            before_sleep=lambda state: logger.warning(
                "%s: временная ошибка, попытка %d/%d: %r",
                service,
                state.attempt_number,
                self._attempts,
                state.outcome.exception() if state.outcome else None,
            ),
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await fn()
        except RetryError as exc:
            cause = exc.last_attempt.exception()
            msg = f"{service} временно недоступен, попробуйте позже"
            raise ExternalServiceUnavailableError(msg) from cause
        msg = f"{service}: повторы завершились без результата"
        raise RuntimeError(msg)  # pragma: no cover
