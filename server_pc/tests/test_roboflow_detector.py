import sys
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
from PIL import Image
from io import BytesIO

from shared.schemas.analysis_result import BoundingBox, Detection, DetectionResult
from server_pc.app.roboflow_detector import ServerRoboflowWorkflowDetector


class FakeRun:
    def as_detection_result(self):
        return DetectionResult(
            detections=[
                Detection(
                    categoryCode="VE",
                    confidence=0.9,
                    boundingBox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
                )
            ],
            image_width=0,
            image_height=0,
            frame_count=1,
        )


class TestServerRoboflowWorkflowDetector(unittest.TestCase):
    def setUp(self):
        timer_patcher = patch(
            "server_pc.app.roboflow_detector.observe_inference_duration",
            return_value=nullcontext(),
        )
        self.observe_duration = timer_patcher.start()
        self.addCleanup(timer_patcher.stop)

    @patch("server_pc.app.roboflow_detector.run_evn_object_detection_workflow")
    def test_detect_image_uses_server_workflow_client(self, run_workflow):
        run_workflow.return_value = [FakeRun()]
        detector = ServerRoboflowWorkflowDetector(api_key="test-key", timeout_seconds=7)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector.detect_image(image)

        self.assertEqual(detector.model_name, "Roboflow Workflow")
        self.assertEqual(result.image_width, 640)
        self.assertEqual(result.image_height, 480)
        self.assertEqual(len(result.detections), 1)
        run_workflow.assert_called_once()
        self.assertEqual(run_workflow.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(run_workflow.call_args.kwargs["timeout_seconds"], 7)
        self.observe_duration.assert_called_once_with("roboflow")

    @patch("server_pc.app.roboflow_detector.run_evn_object_detection_workflow")
    def test_detect_image_url_uses_workflow_url_input(self, run_workflow):
        run_workflow.return_value = [FakeRun()]
        detector = ServerRoboflowWorkflowDetector(api_key="test-key", timeout_seconds=7)

        result = detector.detect_image_url("https://example.com/image.jpg")

        self.assertEqual(len(result.detections), 1)
        run_workflow.assert_called_once()
        self.assertEqual(run_workflow.call_args.args[0], "https://example.com/image.jpg")
        self.assertEqual(run_workflow.call_args.kwargs["api_key"], "test-key")
        self.observe_duration.assert_called_once_with("roboflow")

    @patch("server_pc.app.roboflow_detector.run_evn_object_detection_workflow")
    def test_detect_image_bytes_uses_pil_without_opencv_decode(self, run_workflow):
        run_workflow.return_value = [FakeRun()]
        detector = ServerRoboflowWorkflowDetector(api_key="test-key", timeout_seconds=7)
        buffer = BytesIO()
        Image.new("RGB", (320, 240), color="white").save(buffer, format="JPEG")

        result = detector.detect_image_bytes(buffer.getvalue())

        self.assertEqual(result.image_width, 320)
        self.assertEqual(result.image_height, 240)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(run_workflow.call_args.kwargs["api_key"], "test-key")
        self.observe_duration.assert_called_once_with("roboflow")

    def test_module_import_does_not_load_cv2(self):
        sys.modules.pop("cv2", None)
        import server_pc.app.roboflow_detector  # noqa: F401

        self.assertNotIn("cv2", sys.modules)


if __name__ == "__main__":
    unittest.main()
