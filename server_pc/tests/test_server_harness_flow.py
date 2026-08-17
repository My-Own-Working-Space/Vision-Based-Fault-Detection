import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from server_pc.app.analysis_runner import HarnessAnalysisRunner
from server_pc.app.consumer import create_server_consumer


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_fixture_image() -> bytes:
    """Create a self-contained JPEG fixture without relying on ignored datasets."""
    buffer = BytesIO()
    Image.new("RGB", (64, 48), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


FIXTURE_IMAGE_BYTES = make_fixture_image()


def make_settings(tmpdir):
    return SimpleNamespace(
        callback_base_url="http://backend.local",
        media_download_timeout=10,
        media_max_size_bytes=1024 * 1024,
        allow_private_ips=True,
        ai_service_key="test-service-key",
        callback_max_retries=0,
        callback_retry_base_delay=0,
        callback_retry_max_delay=0,
        callback_timeout=5,
        restrict_callback_to_base_url=True,
        rabbitmq_host="",
        rabbitmq_port=5672,
        rabbitmq_user="guest",
        rabbitmq_pass="guest",
        server_queue_name="server",
        rabbitmq_exchange="ai.analysis",
        edge_queue_name="edge",
        dead_letter_exchange="dlx",
        dead_letter_queue="dlq",
        rabbitmq_heartbeat=600,
        rabbitmq_prefetch_count=1,
        result_exchange="identity-exchange",
        result_routing_key="identity.event.aianalysisresultevent",
        result_queue_name="ai.analysis.result",
        result_retry_queue_name="ai.analysis.result.retry",
        result_dead_letter_queue="ai.analysis.result.dead-letter",
        message_retry_limit=3,
        retry_delay_ms=1000,
        artifact_dir=str(Path(tmpdir) / "artifacts"),
        artifact_public_base_url="http://localhost:8002",
        artifact_url_path="/artifacts",
        artifact_jpeg_quality=90,
        server_port=8002,
        harness_checkpoint_dir=str(Path(tmpdir) / "checkpoints"),
        harness_workflow_ref="fake://evn-object-detection",
    )


class FakeChannel:
    def __init__(self):
        self.acked = []
        self.nacked = []
        self.published = []

    def basic_ack(self, delivery_tag):
        self.acked.append(delivery_tag)

    def basic_nack(self, delivery_tag, requeue=False):
        self.nacked.append((delivery_tag, requeue))

    def confirm_delivery(self):
        return True

    def basic_publish(self, exchange, routing_key, body, properties, mandatory=False):
        self.published.append(
            {
                "exchange": exchange,
                "routing_key": routing_key,
                "body": body,
                "properties": properties,
            }
        )
        return True


class ServerHarnessFlowTests(unittest.TestCase):
    def test_harness_runner_returns_callback_compatible_detection_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(tmpdir) / "checkpoints",
            )

            output = runner.analyze_media(
                file_bytes=FIXTURE_IMAGE_BYTES,
                extension=".jpg",
                media_type="Image",
            )

            self.assertEqual(output.model_name, "HarnessRuntime")
            self.assertTrue(output.harness_run_id.startswith("run-"))
            self.assertTrue(Path(output.harness_checkpoint_path).exists())
            self.assertEqual(output.detection_result.frame_count, 1)
            self.assertGreater(output.detection_result.image_width, 0)
            self.assertGreater(output.detection_result.image_height, 0)
            self.assertGreaterEqual(len(output.detection_result.detections), 1)

    @patch("shared.messaging.result_publisher.publish_analysis_result")
    @patch("shared.messaging.rabbitmq_client.RabbitMQClient")
    @patch("shared.services.media_downloader.download_media")
    def test_rest_api_background_flow_uses_harness_runner_and_callback_contract(
        self,
        download_media,
        rabbitmq_client,
        publish_analysis_result,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(tmpdir)
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(settings.harness_checkpoint_dir),
            )
            download_media.return_value = (FIXTURE_IMAGE_BYTES, ".jpg")

            with patch.dict(
                os.environ,
                {
                    "AI_SERVICE_KEY": "test-service-key",
                    "AI_ARTIFACT_DIR": settings.artifact_dir,
                    "HARNESS_CHECKPOINT_DIR": settings.harness_checkpoint_dir,
                },
            ):
                from server_pc.app import main as server_main

                old_settings = server_main.settings
                old_runner = server_main.analysis_runner
                server_main.settings = settings
                server_main.analysis_runner = runner
                fake_channel = FakeChannel()
                rabbitmq_client.return_value.connect.return_value = fake_channel
                try:
                    payload = server_main.AnalyzePayload(
                        requestId="req-rest-1",
                        mediaId="media-rest-1",
                        fileUrl="/uploads/image.jpg",
                        mediaType="Image",
                        correlationId="corr-rest-1",
                    )
                    server_main._run_analysis(payload)
                finally:
                    server_main.settings = old_settings
                    server_main.analysis_runner = old_runner

            publish_analysis_result.assert_called_once()
            result_event = publish_analysis_result.call_args.args[1]
            self.assertEqual(result_event.analysis_id, "req-rest-1")
            self.assertEqual(result_event.media_id, "media-rest-1")
            self.assertEqual(result_event.model_name, "HarnessRuntime")
            self.assertGreaterEqual(len(result_event.results), 1)

    @patch("shared.messaging.result_publisher.publish_analysis_result")
    @patch("shared.messaging.rabbitmq_client.RabbitMQClient")
    @patch("shared.services.media_downloader.download_media")
    def test_api_analyze_endpoint_schedules_harness_flow(
        self,
        download_media,
        rabbitmq_client,
        publish_analysis_result,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(tmpdir)
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(settings.harness_checkpoint_dir),
            )
            download_media.return_value = (FIXTURE_IMAGE_BYTES, ".jpg")

            with patch.dict(
                os.environ,
                {
                    "AI_SERVICE_KEY": "test-service-key",
                    "AI_ARTIFACT_DIR": settings.artifact_dir,
                    "HARNESS_CHECKPOINT_DIR": settings.harness_checkpoint_dir,
                },
            ):
                from fastapi.testclient import TestClient
                from server_pc.app import main as server_main

                old_settings = server_main.settings
                old_runner = server_main.analysis_runner
                old_ready = server_main._ready
                server_main.settings = settings
                server_main.analysis_runner = runner
                server_main._ready = True
                fake_channel = FakeChannel()
                rabbitmq_client.return_value.connect.return_value = fake_channel
                try:
                    client = TestClient(server_main.app)
                    response = client.post(
                        "/api/analyze",
                        json={
                            "requestId": "req-api-1",
                            "mediaId": "media-api-1",
                        "fileUrl": "/uploads/image.jpg",
                        "mediaType": "Image",
                        "preferredModel": "SERVER",
                        "correlationId": "corr-api-1",
                    },
                )
                finally:
                    server_main.settings = old_settings
                    server_main.analysis_runner = old_runner
                    server_main._ready = old_ready

            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["requestId"], "req-api-1")
            publish_analysis_result.assert_called_once()
            result_event = publish_analysis_result.call_args.args[1]
            self.assertEqual(result_event.analysis_id, "req-api-1")
            self.assertEqual(result_event.model_name, "HarnessRuntime")

    @patch("server_pc.app.consumer.download_media")
    def test_rabbitmq_consumer_flow_publishes_result_event_and_acks(
        self,
        download_media,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(tmpdir)
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(settings.harness_checkpoint_dir),
            )
            download_media.return_value = (FIXTURE_IMAGE_BYTES, ".jpg")
            callback = create_server_consumer(runner, settings)
            channel = FakeChannel()
            method = SimpleNamespace(delivery_tag="delivery-1")
            body = json.dumps(
                {
                    "requestId": "req-rabbit-1",
                    "mediaId": "media-rabbit-1",
                    "fileUrl": "/uploads/image.jpg",
                    "mediaType": "Image",
                    "preferredModel": "SERVER",
                    "correlationId": "corr-rabbit-1",
                }
            ).encode("utf-8")

            callback(channel, method, None, body)

            self.assertEqual(channel.acked, ["delivery-1"])
            self.assertEqual(channel.nacked, [])
            self.assertEqual(len(channel.published), 1)
            payload = json.loads(channel.published[0]["body"].decode("utf-8"))
            self.assertEqual(payload["analysisId"], "req-rabbit-1")
            self.assertEqual(payload["mediaId"], "media-rabbit-1")
            self.assertEqual(payload["modelName"], "HarnessRuntime")
            self.assertGreaterEqual(len(payload["results"]), 1)

    @patch("server_pc.app.consumer.publish_analysis_result")
    @patch("server_pc.app.consumer.download_media")
    def test_publish_failure_does_not_ack_request(
        self,
        download_media,
        publish_analysis_result,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(tmpdir)
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(settings.harness_checkpoint_dir),
            )
            download_media.return_value = (FIXTURE_IMAGE_BYTES, ".jpg")
            from shared.messaging.result_publisher import ResultPublishError
            publish_analysis_result.side_effect = ResultPublishError("publish failed")
            callback = create_server_consumer(runner, settings)
            channel = FakeChannel()
            method = SimpleNamespace(delivery_tag="delivery-2")
            properties = SimpleNamespace(headers={})
            body = json.dumps(
                {
                    "requestId": "req-rabbit-2",
                    "mediaId": "media-rabbit-2",
                    "fileUrl": "/uploads/image.jpg",
                    "mediaType": "Image",
                    "preferredModel": "SERVER",
                    "correlationId": "corr-rabbit-2",
                }
            ).encode("utf-8")

            callback(channel, method, properties, body)

            self.assertEqual(channel.acked, [])
            self.assertEqual(channel.nacked, [("delivery-2", False)])


if __name__ == "__main__":
    unittest.main()
