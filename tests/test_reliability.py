"""Tests for context_harness/reliability.py — Skill 4"""
import asyncio
import pytest
from context_harness.reliability import (
    RetryPolicy, RetryExhausted,
    CircuitBreaker, CircuitState, CircuitOpenError,
    ResilientCaller, with_timeout, with_fallback,
)


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

async def test_retry_succeeds_on_first_attempt():
    policy = RetryPolicy(max_attempts=3, base_delay=0.0)
    calls = []

    async def fn():
        calls.append(1)
        return "ok"

    result = await policy.execute(fn)
    assert result == "ok"
    assert len(calls) == 1


async def test_retry_succeeds_after_transient_failures():
    policy = RetryPolicy(max_attempts=4, base_delay=0.0,
                         retryable_exceptions=(ConnectionError,))
    attempts = [0]

    async def fn():
        attempts[0] += 1
        if attempts[0] < 3:
            raise ConnectionError("transient")
        return "recovered"

    decorated = policy.retrying(lambda e: isinstance(e, ConnectionError))
    result = await decorated(fn)
    assert result == "recovered"
    assert attempts[0] == 3


async def test_retry_raises_after_max_attempts():
    policy = RetryPolicy(max_attempts=3, base_delay=0.0,
                         retryable_exceptions=(ConnectionError,))

    async def always_fail():
        raise ConnectionError("always fails")

    with pytest.raises(RetryExhausted):
        await policy.execute(always_fail)


async def test_retry_does_not_catch_non_retryable():
    policy = RetryPolicy(max_attempts=3, base_delay=0.0,
                         retryable_exceptions=(ConnectionError,))

    async def fn():
        raise ValueError("not retryable")

    decorated = policy.retrying(lambda e: isinstance(e, ConnectionError))
    with pytest.raises(ValueError):
        await decorated(fn)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

async def test_circuit_starts_closed():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
    assert cb.state == CircuitState.CLOSED


async def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

    async def fail():
        raise ConnectionError("fail")

    for _ in range(3):
        with pytest.raises(ConnectionError):
            await cb.call(fail)

    assert cb.state == CircuitState.OPEN


async def test_circuit_open_raises_immediately():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

    async def fail():
        raise ConnectionError("fail")

    for _ in range(2):
        with pytest.raises(ConnectionError):
            await cb.call(fail)

    with pytest.raises(CircuitOpenError):
        await cb.call(fail)


async def test_circuit_reset_closes():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

    async def fail():
        raise ConnectionError()

    for _ in range(2):
        with pytest.raises(Exception):
            await cb.call(fail)

    cb.reset()
    assert cb.state == CircuitState.CLOSED


async def test_circuit_success_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

    async def fail():
        raise ConnectionError()

    async def succeed():
        return "ok"

    for _ in range(2):
        with pytest.raises(ConnectionError):
            await cb.call(fail)

    await cb.call(succeed)
    assert cb._failure_count == 0


def test_circuit_stats_keys():
    cb = CircuitBreaker()
    stats = cb.stats()
    assert "state" in stats
    assert "failure_count" in stats
    assert "success_count" in stats


# ---------------------------------------------------------------------------
# with_timeout
# ---------------------------------------------------------------------------

async def test_with_timeout_succeeds():
    async def fast():
        return "done"

    result = await with_timeout(fast, timeout_seconds=5.0)
    assert result == "done"


async def test_with_timeout_raises_on_slow():
    async def slow():
        await asyncio.sleep(10)
        return "never"

    with pytest.raises(asyncio.TimeoutError):
        await with_timeout(slow, timeout_seconds=0.01)


# ---------------------------------------------------------------------------
# with_fallback
# ---------------------------------------------------------------------------

async def test_fallback_not_called_on_success():
    async def primary():
        return "primary"

    async def fallback():
        return "fallback"

    result = await with_fallback(primary, fallback)
    assert result == "primary"


async def test_fallback_called_on_failure():
    async def primary():
        raise ConnectionError("down")

    async def fallback():
        return "cached"

    result = await with_fallback(primary, fallback, caught_exceptions=(ConnectionError,))
    assert result == "cached"


async def test_fallback_propagates_uncaught_exception():
    async def primary():
        raise ValueError("not caught")

    async def fallback():
        return "should not reach"

    with pytest.raises(ValueError):
        await with_fallback(primary, fallback, caught_exceptions=(ConnectionError,))


# ---------------------------------------------------------------------------
# ResilientCaller
# ---------------------------------------------------------------------------

async def test_resilient_caller_success():
    cb = CircuitBreaker()
    caller = ResilientCaller(
        retry=RetryPolicy(max_attempts=2, base_delay=0.0),
        circuit_breaker=cb,
        timeout_seconds=5.0,
    )

    async def fn():
        return 42

    result = await caller.call(fn)
    assert result == 42


def test_resilient_caller_circuit_stats():
    cb = CircuitBreaker()
    caller = ResilientCaller(circuit_breaker=cb)
    stats = caller.circuit_stats()
    assert stats["state"] == "closed"
