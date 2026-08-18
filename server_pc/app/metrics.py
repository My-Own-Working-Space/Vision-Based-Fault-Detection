"""Prometheus metrics exposed by the Server PC runtime."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


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


def metrics_response():
    """Return the current registry in Prometheus exposition format."""
    return generate_latest(), CONTENT_TYPE_LATEST
