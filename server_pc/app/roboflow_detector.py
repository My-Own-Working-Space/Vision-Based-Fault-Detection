"""
ServerRoboflowWorkflowDetector — hosted Roboflow Workflow inference.

Implements the shared Detector protocol for the server_pc runtime while using
Roboflow Serverless instead of local RF-DETR weights.
"""

from io import BytesIO
from typing import Any, List

from PIL import Image

from shared.schemas.analysis_result import Detection, DetectionResult
from shared.services.roboflow_workflow_client import run_evn_object_detection_workflow
from shared.utils.bbox import calculate_iou
from shared.utils.logging import get_logger
from server_pc.app.metrics import observe_inference_duration

logger = get_logger("server_roboflow_detector")


class ServerRoboflowWorkflowDetector:
    """Detector implementation backed by the published EVN Roboflow Workflow."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 30,
        max_retries: int = 2,
        retry_base_delay: float = 1.0,
        frame_sample_interval: int = 5,
        max_frames_per_video: int = 500,
        dedup_iou_threshold: float = 0.4,
        max_detections_per_class: int = 10,
        api_url: str = "https://serverless.roboflow.com",
        workspace_name: str = "les-workspace-ijdwd",
        workflow_id: str = (
            "evn-object-detection-vevn-object-detection-cnyo0-2-yolo11n-t1-logic"
        ),
    ):
        if not api_key:
            raise ValueError(
                "ROBOFLOW_API_KEY must be configured for Roboflow inference"
            )

        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._frame_sample_interval = frame_sample_interval
        self._max_frames_per_video = max_frames_per_video
        self._dedup_iou_threshold = dedup_iou_threshold
        self._max_detections_per_class = max_detections_per_class
        self._model_name = "Roboflow Workflow"
        self._api_url = api_url
        self._workspace_name = workspace_name
        self._workflow_id = workflow_id
        self._model_version = workflow_id

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def detect_image(self, image: Any) -> DetectionResult:
        """Run hosted Roboflow Workflow inference on a single image array."""
        h, w = image.shape[:2]
        with observe_inference_duration("roboflow"):
            runs = run_evn_object_detection_workflow(
                image,
                api_key=self._api_key,
                api_url=self._api_url,
                workspace_name=self._workspace_name,
                workflow_id=self._workflow_id,
                timeout_seconds=self._timeout_seconds,
                max_retries=self._max_retries,
                retry_base_delay=self._retry_base_delay,
            )
        if not runs:
            return DetectionResult(
                detections=[], image_width=w, image_height=h, frame_count=1
            )

        result = runs[0].as_detection_result()
        if result.image_width <= 0 or result.image_height <= 0:
            result = DetectionResult(
                detections=result.detections,
                image_width=w,
                image_height=h,
                frame_count=1,
            )

        logger.info(
            f"Roboflow image inference: {len(result.detections)} "
            f"detections ({w}x{h})",
            extra={"event": "inference_image_complete"},
        )
        return result


    def detect_image_url(self, image_url: str) -> DetectionResult:
        """Run hosted Roboflow Workflow inference directly from an HTTPS URL."""
        with observe_inference_duration("roboflow"):
            runs = run_evn_object_detection_workflow(
                image_url,
                api_key=self._api_key,
                api_url=self._api_url,
                workspace_name=self._workspace_name,
                workflow_id=self._workflow_id,
                timeout_seconds=self._timeout_seconds,
                max_retries=self._max_retries,
                retry_base_delay=self._retry_base_delay,
            )
        if not runs:
            return DetectionResult(
                detections=[], image_width=0, image_height=0, frame_count=1
            )

        result = runs[0].as_detection_result()
        logger.info(
            f"Roboflow image inference: {len(result.detections)} detections "
            f"({result.image_width}x{result.image_height})",
            extra={"event": "inference_image_complete"},
        )
        return result

    def detect_image_bytes(self, image_bytes: bytes) -> DetectionResult:
        """Run hosted Roboflow Workflow inference on encoded image bytes.

        This path avoids importing OpenCV in production Roboflow deployments,
        which keeps the lightweight Docker image independent from libGL.
        """
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            w, h = image.size
            with observe_inference_duration("roboflow"):
                runs = run_evn_object_detection_workflow(
                    image,
                    api_key=self._api_key,
                    timeout_seconds=self._timeout_seconds,
                    max_retries=self._max_retries,
                    retry_base_delay=self._retry_base_delay,
                )

        if not runs:
            return DetectionResult(
                detections=[], image_width=w, image_height=h, frame_count=1
            )

        result = runs[0].as_detection_result()
        if result.image_width <= 0 or result.image_height <= 0:
            result = DetectionResult(
                detections=result.detections,
                image_width=w,
                image_height=h,
                frame_count=1,
            )

        logger.info(
            f"Roboflow image inference: {len(result.detections)} "
            f"detections ({w}x{h})",
            extra={"event": "inference_image_complete"},
        )
        return result

    def detect_video(self, video_path: str) -> DetectionResult:
        """Run hosted Roboflow Workflow inference on sampled video frames."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = round(total_frames / fps, 3) if fps and total_frames else None

        all_detections: List[Detection] = []
        frame_idx = 0
        processed_count = 0

        try:
            while processed_count < self._max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self._frame_sample_interval == 0:
                    timestamp_ms = int((frame_idx / fps) * 1000)
                    frame_result = self.detect_image(frame)
                    for det in frame_result.detections:
                        det.timestamp_ms = timestamp_ms
                        det.frame_index = frame_idx
                        all_detections.append(det)
                    processed_count += 1

                frame_idx += 1
        finally:
            cap.release()

        deduped = self._deduplicate(all_detections)
        logger.info(
            f"Roboflow video inference: {len(deduped)} detections "
            f"(from {processed_count} frames, {frame_idx} total)",
            extra={"event": "inference_video_complete"},
        )
        return DetectionResult(
            detections=deduped,
            image_width=w,
            image_height=h,
            frame_count=processed_count,
            fps=round(fps, 3),
            duration=duration,
        )

    def _deduplicate(self, detections: List[Detection]) -> List[Detection]:
        grouped: dict = {}
        for det in detections:
            grouped.setdefault(det.category_code, []).append(det)

        result: List[Detection] = []
        for category, items in grouped.items():
            items.sort(key=lambda d: d.confidence, reverse=True)
            kept: List[Detection] = []
            for det in items:
                if len(kept) >= self._max_detections_per_class:
                    break
                box = det.bounding_box.model_dump()
                if any(
                    calculate_iou(box, existing.bounding_box.model_dump())
                    > self._dedup_iou_threshold
                    for existing in kept
                ):
                    continue
                kept.append(det)
            result.extend(kept)

        return result
