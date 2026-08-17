"""
RabbitMQ consumer for server_pc runtime.

Consumes analysis jobs from the server queue, runs the configured server
analysis runner, and publishes result events back to RabbitMQ.
"""

import json
import time
import threading
from datetime import datetime, timezone
from json import JSONDecodeError

import pika
from shared.schemas.analysis_request import AnalysisRequest, PreferredModel
from shared.services.media_downloader import (
    download_media,
    resolve_media_url,
    DownloadError,
)
from shared.services.result_mapper import map_success_result, map_failure_result
from shared.services.result_event_mapper import map_result_to_event
from shared.messaging.rabbitmq_client import (
    consume_with_reconnect,
    create_media_dispatcher,
)
from shared.messaging.result_publisher import (
    publish_analysis_result,
    ResultPublishError,
)
from shared.utils.logging import (
    get_logger,
    set_correlation_context,
    clear_correlation_context,
)

logger = get_logger("server_consumer")


def create_server_consumer(analysis_runner, settings):
    """Create a RabbitMQ message handler for the server runtime.

    Args:
        analysis_runner: Initialized server analysis runner.
        settings: ServerSettings instance.

    Returns:
        Message callback function.
    """

    def on_message(ch, method, properties, body):
        """Handle incoming analysis request messages."""
        start_time = time.monotonic()
        retry_count = _get_retry_count(properties)

        try:
            payload = json.loads(body.decode("utf-8"))
            request = AnalysisRequest(**payload)

            # Set logging context
            set_correlation_context(
                correlation_id=request.correlation_id,
                request_id=request.request_id,
            )

            logger.info(
                f"Received server analysis job: {request.request_id}",
                extra={"event": "job_received"},
            )

            # Validate routing — reject edge-targeted jobs
            if request.preferred_model and PreferredModel.is_edge(
                request.preferred_model
            ):
                logger.warning(
                    f"Job {request.request_id} has preferredModel="
                    f"{request.preferred_model}, rejecting from server queue",
                    extra={"event": "job_rejected_wrong_runtime"},
                )
                # Reject — do not requeue (DLQ will catch it)
                ch.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=False
                )
                return

            # Run inference through the configured analysis runner.
            try:
                if _can_analyze_image_url(analysis_runner, request.media_type.value):
                    media_url = resolve_media_url(
                        file_url=request.file_url,
                        base_url=settings.callback_base_url,
                        allow_private_ips=settings.allow_private_ips,
                    )
                    if not media_url.startswith("https://"):
                        raise DownloadError("Roboflow URL inputs must use https")
                    output = analysis_runner.analyze_url(
                        file_url=media_url,
                        media_type=request.media_type.value,
                    )
                else:
                    file_bytes, ext = download_media(
                        file_url=request.file_url,
                        base_url=settings.callback_base_url,
                        timeout=settings.media_download_timeout,
                        max_size_bytes=settings.media_max_size_bytes,
                        allow_private_ips=settings.allow_private_ips,
                    )
                    output = analysis_runner.analyze_media(
                        file_bytes=file_bytes,
                        extension=ext,
                        media_type=request.media_type.value,
                        request_id=request.request_id,
                        artifact_dir=settings.artifact_dir,
                        public_base_url=(
                            settings.artifact_public_base_url
                            or f"http://localhost:{settings.server_port}"
                        ),
                        artifact_url_path=settings.artifact_url_path,
                        jpeg_quality=settings.artifact_jpeg_quality,
                    )
            except DownloadError as e:
                result = _build_failure_result(
                    request, "MEDIA_DOWNLOAD_FAILED", str(e)
                )
                _publish_result_and_finalize(ch, method.delivery_tag, result, settings)
                return
            except Exception as e:
                logger.error(
                    f"Model inference failed for job {request.request_id}: {e}",
                    exc_info=True,
                    extra={
                        "event": "job_inference_failed",
                        "error_code": "MODEL_INFERENCE_FAILED",
                    },
                )
                result = _build_failure_result(
                    request, "MODEL_INFERENCE_FAILED", str(e)
                )
                _publish_result_and_finalize(ch, method.delivery_tag, result, settings)
                return

            # Build and publish success result
            processing_time_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = map_success_result(
                request_id=request.request_id,
                media_id=request.media_id,
                detection_result=output.detection_result,
                model_name=output.model_name,
                model_version=output.model_version,
                processing_time_ms=processing_time_ms,
                device_profile="server",
                asset_id=request.asset_id,
                mission_id=request.mission_id,
                correlation_id=request.correlation_id,
                image_url=request.file_url,
            )

            if output.harness_run_id:
                result.raw_result["harnessRunId"] = output.harness_run_id
            if output.harness_checkpoint_path:
                result.raw_result["harnessCheckpointPath"] = output.harness_checkpoint_path

            try:
                _publish_result_and_finalize(ch, method.delivery_tag, result, settings)
                logger.info(
                    f"Job {request.request_id} completed successfully "
                    f"in {processing_time_ms}ms",
                    extra={
                        "event": "job_completed",
                        "duration_ms": processing_time_ms,
                    },
                )
            except ResultPublishError:
                logger.error(
                    f"Result publish failed for job {request.request_id}",
                    extra={"event": "job_result_publish_failed"},
                )
                ch.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=False
                )

        except JSONDecodeError as e:
            logger.error(
                f"Invalid JSON in message: {e}",
                extra={"event": "job_invalid_json"},
            )
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        except Exception as e:
            logger.error(
                f"Unexpected error processing message: {e}",
                exc_info=True,
                extra={"event": "job_unexpected_error"},
            )
            _retry_or_dead_letter(
                ch, method.delivery_tag, body, properties, settings, retry_count
            )

        finally:
            clear_correlation_context()

    return on_message


def _can_analyze_image_url(analysis_runner, media_type: str) -> bool:
    return media_type.lower() == "image" and hasattr(analysis_runner, "analyze_url")


def _build_failure_result(request, error_code, error_message):
    return map_failure_result(
        request_id=request.request_id,
        media_id=request.media_id,
        mission_id=request.mission_id,
        asset_id=request.asset_id,
        correlation_id=request.correlation_id,
        error_code=error_code,
        error_message=error_message,
    )


def _publish_result_and_finalize(ch, delivery_tag, result, settings):
    result_event = map_result_to_event(result)
    publish_analysis_result(
        ch,
        result_event,
        routing_key=settings.result_routing_key,
        exchange=settings.result_exchange,
    )

    ch.basic_ack(delivery_tag=delivery_tag)


def _get_retry_count(properties) -> int:
    headers = getattr(properties, "headers", None) or {}
    raw = headers.get("x-retry-count", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _retry_or_dead_letter(ch, delivery_tag, body, properties, settings, retry_count):
    if retry_count >= settings.message_retry_limit:
        ch.basic_nack(delivery_tag=delivery_tag, requeue=False)
        return

    try:
        headers = dict(getattr(properties, "headers", None) or {})
        headers["x-retry-count"] = retry_count + 1
        ch.basic_publish(
            exchange="",
            routing_key=settings.server_queue_name,
            body=body,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
                headers=headers,
                correlation_id=getattr(properties, "correlation_id", None),
                message_id=getattr(properties, "message_id", None),
                timestamp=int(datetime.now(timezone.utc).timestamp()),
            ),
        )
        ch.basic_ack(delivery_tag=delivery_tag)
    except Exception:
        ch.basic_nack(delivery_tag=delivery_tag, requeue=False)


def start_server_consumer(analysis_runner, settings):
    """Start ingress, image, and video consumers in independent threads.

    Args:
        analysis_runner: Initialized server analysis runner.
        settings: ServerSettings instance.
    """
    work_callback = create_server_consumer(analysis_runner, settings)
    dispatch_callback = create_media_dispatcher(
        settings.server_image_queue_name, settings.server_video_queue_name
    )

    common = dict(
        host=settings.rabbitmq_host,
        port=settings.rabbitmq_port,
        user=settings.rabbitmq_user,
        password=settings.rabbitmq_pass,
        exchange=settings.rabbitmq_exchange,
        edge_queue=settings.edge_queue_name,
        server_queue=settings.server_queue_name,
        edge_image_queue=settings.edge_image_queue_name,
        edge_video_queue=settings.edge_video_queue_name,
        server_image_queue=settings.server_image_queue_name,
        server_video_queue=settings.server_video_queue_name,
        dlx_exchange=settings.dead_letter_exchange,
        dlq_queue=settings.dead_letter_queue,
        heartbeat=settings.rabbitmq_heartbeat,
        prefetch_count=settings.rabbitmq_prefetch_count,
    )

    def run(queue_name, callback):
        consume_with_reconnect(
            **common,
            queue_name=queue_name,
            callback=callback,
        )

    consumers = (
        (settings.server_queue_name, dispatch_callback, "server-dispatcher"),
        (settings.server_image_queue_name, work_callback, "server-image-consumer"),
        (settings.server_video_queue_name, work_callback, "server-video-consumer"),
    )
    threads = []
    for queue_name, callback, name in consumers:
        thread = threading.Thread(
            target=run, args=(queue_name, callback), daemon=True, name=name
        )
        thread.start()
        threads.append(thread)
        logger.info(
            "Server RabbitMQ consumer started",
            extra={"event": "consumer_started", "queue": queue_name},
        )
    return threads
