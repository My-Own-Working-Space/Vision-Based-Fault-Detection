"""Prometheus metrics and safe recording helpers for the Server PC runtime."""

import logging
import time
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

logger = logging.getLogger(__name__)

_JOB_STATUSES = frozenset({"success", "failed", "rejected"})
_MEDIA_TYPES = frozenset({"Image", "Video", "unknown"})
_INFERENCE_BACKENDS = frozenset({"rfdetr", "transformers", "roboflow", "harness"})

AI_JOBS_TOTAL = Counter(
    "ai_jobs_total",
    "Total number of AI jobs processed.",
    ("status", "media_type"),
)

AI_JOB_DURATION = Histogram(
    "ai_job_duration_seconds",
    "End-to-end AI job processing duration in seconds.",
    ("media_type",),
)

AI_INFERENCE_DURATION = Histogram(
    "ai_inference_duration_seconds",
    "AI model inference duration in seconds.",
    ("backend",),
)


def record_job(status: str, media_type: str) -> None:
    """Increment a job outcome without allowing metrics to affect processing."""
    try:
        safe_status = status if status in _JOB_STATUSES else "failed"
        safe_media_type = media_type if media_type in _MEDIA_TYPES else "unknown"
        AI_JOBS_TOTAL.labels(
            status=safe_status, media_type=safe_media_type
        ).inc()
    except Exception:
        logger.warning("Failed to record AI job metric", exc_info=True)


def observe_job_duration(media_type: str, duration_seconds: float) -> None:
    """Observe end-to-end job duration with bounded labels."""
    try:
        safe_media_type = media_type if media_type in _MEDIA_TYPES else "unknown"
        AI_JOB_DURATION.labels(media_type=safe_media_type).observe(duration_seconds)
    except Exception:
        logger.warning("Failed to record AI job duration metric", exc_info=True)


@contextmanager
def observe_inference_duration(backend: str):
    """Measure an inference boundary without changing its error behavior."""
    start_time = time.monotonic()
    try:
        yield
    finally:
        try:
            if backend not in _INFERENCE_BACKENDS:
                logger.warning("Ignored unknown AI inference backend label")
            else:
                AI_INFERENCE_DURATION.labels(backend=backend).observe(
                    time.monotonic() - start_time
                )
        except Exception:
            logger.warning("Failed to record AI inference metric", exc_info=True)


def metrics_response():
    """Return the current registry in Prometheus exposition format."""
    return generate_latest(), CONTENT_TYPE_LATEST
