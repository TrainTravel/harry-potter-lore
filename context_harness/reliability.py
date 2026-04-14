"""
Reliability Layer — Skill 4: Reliability Engineering
=====================================================
The IBM video's point: LLM API calls fail. Networking fails. Rate limits hit.
An agent without retry/timeout/circuit-breaker logic will break in production
on the first transient error.

This module implements the standard reliability toolkit:
  - RetryPolicy  — exponential backoff with jitter, configurable exceptions
  - CircuitBreaker — closed/open/half-open state machine
  - with_timeout  — async timeout wrapper
  - with_fallback — run a fallback coroutine if the primary fails

Design principle:
  "Treat every external call as if it will fail. Design for recovery first."
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, List, Optional, Tuple, Type


# ---------------------------------------------------------------------------
# Retry Policy
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """
    Exponential backoff with full jitter.

    Jitter prevents thundering-herd when many agents hit the same API
    at the same time after a shared outage.

    Usage:
        policy = RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0)
        result = await policy.execute(my_async_fn, arg1, arg2)
    """

    max_attempts: int = 4
    base_delay: float = 1.0       # seconds
    max_delay: float = 30.0       # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError, TimeoutError, OSError,
    )

    async def execute(self, fn: Callable[..., Coroutine], *args, **kwargs) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await fn(*args, **kwargs)
            except self.retryable_exceptions as exc:
                last_exc = exc
                if attempt == self.max_attempts:
                    break
                delay = self._delay(attempt)
                await asyncio.sleep(delay)
            except Exception:
                raise   # non-retryable: re-raise immediately

        raise RetryExhausted(
            f"All {self.max_attempts} attempts failed. Last error: {last_exc}"
        ) from last_exc

    def retrying(self, retry_on: Callable[[Exception], bool]):
        """
        Return an async callable that retries fn only when retry_on(exc) is True.
        All other exceptions propagate immediately.

        Usage:
            result = await policy.retrying(lambda e: isinstance(e, ConnectionError))(my_fn)
        """
        async def execute(fn: Callable[..., Coroutine], *args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    if not retry_on(exc):
                        raise
                    last_exc = exc
                    if attempt == self.max_attempts:
                        break
                    await asyncio.sleep(self._delay(attempt))
            raise RetryExhausted(
                f"All {self.max_attempts} attempts failed. Last error: {last_exc}"
            ) from last_exc
        return execute

    def _delay(self, attempt: int) -> float:
        """Full-jitter exponential backoff."""
        cap = min(self.max_delay, self.base_delay * (self.exponential_base ** (attempt - 1)))
        return random.uniform(0, cap) if self.jitter else cap


class RetryExhausted(RuntimeError):
    """Raised when all retry attempts are exhausted."""


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    CLOSED    = "closed"      # normal: requests pass through
    OPEN      = "open"        # tripped: requests fail immediately
    HALF_OPEN = "half_open"   # testing: one probe request allowed


class CircuitBreaker:
    """
    Classic circuit breaker pattern.

    States:
      CLOSED    → requests pass through; failures counted
      OPEN      → requests fail immediately (fast-fail, no waiting)
      HALF_OPEN → one probe request passes; success → CLOSED, failure → OPEN

    Usage:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        try:
            result = await cb.call(my_async_fn, arg)
        except CircuitOpenError:
            # fast-fail path: use cached result or return degraded response
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,    # consecutive successes needed to close
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - (self._opened_at or 0) >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    async def call(self, fn: Callable[..., Coroutine], *args, **kwargs) -> Any:
        current = self.state

        if current == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit is OPEN. Failing fast. "
                f"Retrying after {self.recovery_timeout}s from trip."
            )

        try:
            result = await fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._success_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._success_count = 0
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = None

    def stats(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted while the circuit is open."""


# ---------------------------------------------------------------------------
# Timeout wrapper
# ---------------------------------------------------------------------------

async def with_timeout(
    fn: Callable[..., Coroutine],
    *args,
    timeout_seconds: float = 30.0,
    **kwargs,
) -> Any:
    """
    Wrap an async call with a hard timeout.
    Raises asyncio.TimeoutError if the call exceeds the limit.

    Usage:
        result = await with_timeout(call_llm, prompt, timeout_seconds=10.0)
    """
    return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_seconds)


# ---------------------------------------------------------------------------
# Fallback wrapper
# ---------------------------------------------------------------------------

async def with_fallback(
    primary: Callable[..., Coroutine],
    fallback: Callable[..., Coroutine],
    *args,
    caught_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs,
) -> Any:
    """
    Try primary; on any caught exception run fallback instead.

    Useful for: cached response when RAG is down, degraded mode when LLM is slow.

    Usage:
        result = await with_fallback(
            call_live_llm, call_cached_response,
            prompt,
            caught_exceptions=(TimeoutError, ConnectionError),
        )
    """
    try:
        return await primary(*args, **kwargs)
    except caught_exceptions:
        return await fallback(*args, **kwargs)


# ---------------------------------------------------------------------------
# Combined resilient caller
# ---------------------------------------------------------------------------

class ResilientCaller:
    """
    Combines RetryPolicy + CircuitBreaker + timeout in one object.

    This is what you wrap around every external LLM or API call.

    Usage:
        caller = ResilientCaller(
            retry=RetryPolicy(max_attempts=3),
            circuit_breaker=CircuitBreaker(failure_threshold=5),
            timeout_seconds=20.0,
        )
        result = await caller.call(my_llm_fn, prompt)
    """

    def __init__(
        self,
        retry: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._retry = retry or RetryPolicy()
        self._cb = circuit_breaker or CircuitBreaker()
        self._timeout = timeout_seconds

    async def call(self, fn: Callable[..., Coroutine], *args, **kwargs) -> Any:
        async def timed_fn(*a, **kw):
            return await with_timeout(fn, *a, timeout_seconds=self._timeout, **kw)

        return await self._retry.execute(
            lambda *a, **kw: self._cb.call(timed_fn, *a, **kw),
            *args, **kwargs,
        )

    def circuit_stats(self) -> dict:
        return self._cb.stats()
