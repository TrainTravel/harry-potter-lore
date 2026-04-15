"""
OpenTelemetry bridge — optional span emission alongside Tracer.event()
======================================================================
Call configure() once at startup to route spans to Jaeger (or any OTLP
collector). If configure() is never called every emit_span() call is a
cheap no-op — no imports, no overhead.

Usage:
    from context_harness.otel_bridge import configure
    configure(service_name="deeptutor", endpoint="http://localhost:4317")
    # From this point every Tracer.event() call also emits an OTel span.

Viewing in Jaeger:
    docker compose -f docker-compose.jaeger.yml up -d
    # then open http://localhost:16686
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level tracer — None until configure() is called.
# Type: opentelemetry.trace.Tracer | None
_otel_tracer = None
_provider = None  # kept alive so the BatchSpanProcessor can flush on shutdown


def configure(
    service_name: str = "deeptutor",
    endpoint: str = "http://localhost:4317",
) -> None:
    """
    Wire up an OTLP gRPC exporter to Jaeger (or any OTel collector).

    Idempotent — safe to call multiple times; only the first call has effect.
    Raises ImportError if opentelemetry-sdk or opentelemetry-exporter-otlp-proto-grpc
    are not installed (both are in requirements.txt).

    Args:
        service_name: Value for the OTel 'service.name' resource attribute.
        endpoint:     gRPC OTLP endpoint, e.g. "http://localhost:4317".
    """
    global _otel_tracer, _provider
    if _otel_tracer is not None:
        return

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    resource = Resource(attributes={SERVICE_NAME: service_name})
    _provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)

    _otel_tracer = trace.get_tracer(service_name)
    logger.info("OTel configured: service=%s endpoint=%s", service_name, endpoint)


def emit_span(
    turn_id: str,
    seq: int,
    kind_value: str,
    payload: Dict[str, Any],
    latency_ms: Optional[float] = None,
) -> None:
    """
    Emit one OTel span for a single TraceEvent.

    The span is created and immediately ended (it represents a discrete event,
    not an ongoing operation). Errors in span emission are logged and swallowed
    — observability must never take down the thing it observes.

    Args:
        turn_id:    The turn's unique ID (becomes a span attribute).
        seq:        Monotonic sequence number within the turn.
        kind_value: String value of EventKind, e.g. "llm_call".
        payload:    Arbitrary key→value pairs from the event. Values are
                    truncated to 256 chars to stay within OTel attribute limits.
        latency_ms: Optional duration already measured by the caller.
    """
    if _otel_tracer is None:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind, StatusCode

        attributes: Dict[str, Any] = {
            "turn.id": turn_id,
            "event.seq": seq,
            "event.kind": kind_value,
        }
        if latency_ms is not None:
            attributes["latency_ms"] = latency_ms
        # Flatten payload — prefix keys to avoid collisions with OTel conventions
        for k, v in payload.items():
            attributes[f"payload.{k}"] = str(v)[:256]

        span = _otel_tracer.start_span(
            name=kind_value,
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        )
        if kind_value == "error":
            span.set_status(StatusCode.ERROR)
        span.end()
    except Exception:
        logger.debug("OTel span emission failed (non-fatal)", exc_info=True)


def is_configured() -> bool:
    """Return True if configure() has been called successfully."""
    return _otel_tracer is not None


def shutdown() -> None:
    """
    Flush and shut down the tracer provider. Call at process exit to ensure
    the BatchSpanProcessor drains its queue before the process terminates.
    """
    if _provider is not None:
        _provider.shutdown()
