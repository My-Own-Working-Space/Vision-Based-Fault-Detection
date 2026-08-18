import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from server_pc.app.analysis_runner import HarnessAnalysisRunner, ServerAnalysisOutput
from server_pc.app.consumer import create_server_consumer
from server_pc.app.detector import ServerRfDetrDetector
from server_pc.app import metrics
from shared.schemas.analysis_result import DetectionResult

from server_pc.tests.test_server_harness_flow import (
    FIXTURE_IMAGE_BYTES,
    FakeChannel,
    REPO_ROOT,
    make_settings,
)


class FakeMetric:
    def __init__(self):
        self.labels_calls = []
        self.observations = []
        self.increments = 0

    def labels(self, **labels):
        self.labels_calls.append(labels)
        return self

    def inc(self):
        self.increments += 1

    def observe(self, value):
        self.observations.append(value)


class MetricsHelpersTests(unittest.TestCase):
    def test_metrics_exposition_registers_all_custom_metrics(self):
        body, content_type = metrics.metrics_response()

        self.assertIn(b"# TYPE ai_jobs_total counter", body)
        self.assertIn(b"# TYPE ai_job_duration_seconds histogram", body)
        self.assertIn(b"# TYPE ai_inference_duration_seconds histogram", body)
        self.assertIn("text/plain", content_type)

    def test_job_labels_are_bounded_and_duration_is_observed(self):
        counter = FakeMetric()
        histogram = FakeMetric()
        with patch.object(metrics, "AI_JOBS_TOTAL", counter), patch.object(
            metrics, "AI_JOB_DURATION", histogram
        ):
            metrics.record_job("value-from-user", "https://sensitive.example/file")
            metrics.observe_job_duration("Video", 1.25)

        self.assertEqual(
            counter.labels_calls, [{"status": "failed", "media_type": "unknown"}]
        )
        self.assertEqual(histogram.labels_calls, [{"media_type": "Video"}])
        self.assertEqual(histogram.observations, [1.25])

    def test_inference_duration_is_observed_with_stable_backend(self):
        histogram = FakeMetric()
        with patch.object(metrics, "AI_INFERENCE_DURATION", histogram):
            with metrics.observe_inference_duration("rfdetr"):
                pass

        self.assertEqual(histogram.labels_calls, [{"backend": "rfdetr"}])
        self.assertEqual(len(histogram.observations), 1)
        self.assertGreaterEqual(histogram.observations[0], 0)

    def test_unknown_inference_backend_is_not_exported(self):
        histogram = FakeMetric()
        with patch.object(metrics, "AI_INFERENCE_DURATION", histogram):
            with metrics.observe_inference_duration("request-specific-value"):
                pass

        self.assertEqual(histogram.labels_calls, [])


class ConsumerMetricsTests(unittest.TestCase):
    def _body(self):
        return json.dumps(
            {
                "requestId": "req-metrics",
                "mediaId": "media-metrics",
                "fileUrl": "/uploads/image.jpg",
                "mediaType": "Image",
                "preferredModel": "SERVER",
            }
        ).encode("utf-8")

    @patch("server_pc.app.consumer.observe_job_duration")
    @patch("server_pc.app.consumer.record_job")
    @patch("server_pc.app.consumer.download_media")
    def test_success_records_counter_and_existing_job_duration(
        self, download_media, record_job, observe_job_duration
    ):
        runner = MagicMock(spec=["analyze_media"])
        runner.analyze_media.return_value = ServerAnalysisOutput(
            detection_result=DetectionResult(
                detections=[], image_width=10, image_height=10, frame_count=1
            ),
            model_name="test-model",
            model_version="1",
        )
        download_media.return_value = (b"image", ".jpg")
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = create_server_consumer(runner, make_settings(tmpdir))
            channel = FakeChannel()
            callback(
                channel,
                SimpleNamespace(delivery_tag="success"),
                SimpleNamespace(headers={}),
                self._body(),
            )

        record_job.assert_called_once_with("success", "Image")
        observe_job_duration.assert_called_once()
        self.assertGreaterEqual(observe_job_duration.call_args.args[1], 0)
        self.assertEqual(channel.acked, ["success"])

    @patch("server_pc.app.consumer.observe_job_duration")
    @patch("server_pc.app.consumer.record_job")
    @patch("server_pc.app.consumer.download_media")
    def test_failed_inference_records_failed_once_without_duration(
        self, download_media, record_job, observe_job_duration
    ):
        runner = MagicMock(spec=["analyze_media"])
        runner.analyze_media.side_effect = RuntimeError("model failed")
        download_media.return_value = (b"image", ".jpg")
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = create_server_consumer(runner, make_settings(tmpdir))
            channel = FakeChannel()
            callback(
                channel,
                SimpleNamespace(delivery_tag="failed"),
                SimpleNamespace(headers={}),
                self._body(),
            )

        record_job.assert_called_once_with("failed", "Image")
        observe_job_duration.assert_not_called()
        self.assertEqual(channel.acked, ["failed"])

    @patch("server_pc.app.consumer.record_job")
    def test_invalid_json_is_rejected_with_unknown_media_type(self, record_job):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = create_server_consumer(MagicMock(), make_settings(tmpdir))
            channel = FakeChannel()
            callback(
                channel,
                SimpleNamespace(delivery_tag="invalid"),
                SimpleNamespace(headers={}),
                b"not-json",
            )

        record_job.assert_called_once_with("rejected", "unknown")
        self.assertEqual(channel.nacked, [("invalid", False)])


class InferenceBoundaryTests(unittest.TestCase):
    @patch("server_pc.app.detector.observe_inference_duration")
    def test_rfdetr_predict_is_measured(self, observe_duration):
        observe_duration.return_value = nullcontext()
        detector = ServerRfDetrDetector.__new__(ServerRfDetrDetector)
        detector._model = MagicMock()
        detector._model.predict.return_value = SimpleNamespace(xyxy=[])
        detector._conf_threshold = 0.3

        detector._detect_rfdetr(np.zeros((8, 8, 3), dtype=np.uint8), 8, 8)

        observe_duration.assert_called_once_with("rfdetr")
        detector._model.predict.assert_called_once()

    @patch("server_pc.app.analysis_runner.observe_inference_duration")
    def test_harness_start_run_is_measured_once(self, observe_duration):
        observe_duration.return_value = nullcontext()
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = HarnessAnalysisRunner(
                repo_root=REPO_ROOT,
                checkpoint_dir=Path(tmpdir) / "checkpoints",
            )
            runner.analyze_media(FIXTURE_IMAGE_BYTES, ".jpg", "Image")

        observe_duration.assert_called_once_with("harness")


if __name__ == "__main__":
    unittest.main()
