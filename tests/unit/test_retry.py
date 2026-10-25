import pytest

from rag.adapters.outbound.retry import RetryPolicy
from rag.domain.exceptions import ExternalServiceUnavailableError


class TransientError(Exception):
    pass


def is_transient(exc: BaseException) -> bool:
    return isinstance(exc, TransientError)


def flaky(failures: int, exc: Exception) -> tuple[list[int], object]:
    calls: list[int] = []

    async def fn() -> str:
        calls.append(1)
        if len(calls) <= failures:
            raise exc
        return "ok"

    return calls, fn


async def test_retries_transient_errors_until_success() -> None:
    calls, fn = flaky(2, TransientError())
    assert await RetryPolicy(attempts=3, max_wait_seconds=0.01).call("svc", fn, is_transient) == "ok"
    assert len(calls) == 3


async def test_exhausted_retries_become_service_unavailable() -> None:
    calls, fn = flaky(10, TransientError())
    with pytest.raises(ExternalServiceUnavailableError) as info:
        await RetryPolicy(attempts=3, max_wait_seconds=0.01).call("Qdrant", fn, is_transient)
    assert len(calls) == 3
    assert isinstance(info.value.__cause__, TransientError)


async def test_permanent_errors_are_not_retried() -> None:
    calls, fn = flaky(10, ValueError("bad request"))
    with pytest.raises(ValueError, match="bad request"):
        await RetryPolicy(attempts=3, max_wait_seconds=0.01).call("svc", fn, is_transient)
    assert len(calls) == 1
