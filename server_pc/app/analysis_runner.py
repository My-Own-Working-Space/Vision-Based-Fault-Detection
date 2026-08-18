"""
Compatibility runners for server_pc analysis execution.

These runners preserve the existing callback contract by returning
shared.schemas.analysis_result.DetectionResult while allowing the server paths
to execute through the provider-independent harness.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from edge.harness.models import LocalImageTrigger, RunResultStatus
from edge.harness.runtime import HarnessRuntime
from shared.schemas.analysis_result import BoundingBox, Detection, DetectionResult
from shared.services.class_mapping import map_class_to_category
from shared.services.media_downloader import save_to_temp_file
from shared.utils.logging import get_logger
from server_pc.app.metrics import observe_inference_duration

logger = get_logger("server_analysis_runner")

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


@dataclass(frozen=True)
class ServerAnalysisOutput:
    detection_result: DetectionResult
    model_name: str
    model_version: str
    harness_run_id: Optional[str] = None
    harness_checkpoint_path: Optional[str] = None


class HarnessAnalysisRunner:
    """Server compatibility adapter that executes image analysis through HarnessRuntime."""

    def __init__(
        self,
        repo_root: Path,
        checkpoint_dir: Path,
        workflow_ref: str = "fake://evn-object-detection",
    ):
        self._repo_root = repo_root
        self._checkpoint_dir = checkpoint_dir
        self._workflow_ref = workflow_ref
        self._model_name = "HarnessRuntime"
        self._model_version = "fake-provider-v1"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def analyze_media(
        self,
        file_bytes: bytes,
        extension: str,
        media_type: str = "Image",
        request_id: str = "",
        artifact_dir: str = "",
        public_base_url: str = "",
        artifact_url_path: str = "/artifacts",
        jpeg_quality: int = 90,
    ) -> ServerAnalysisOutput:
        if _is_video(media_type, extension):
            raise ValueError("Harness backend currently supports image media only")

        temp_path = save_to_temp_file(file_bytes, extension or ".jpg")
        try:
            runtime = HarnessRuntime(
                repo_root=self._repo_root,
                checkpoint_dir=self._checkpoint_dir,
                emit_logs=False,
            )
            with observe_inference_duration("harness"):
                result = runtime.start_run(
                    LocalImageTrigger(
                        image_path=Path(temp_path),
                        workflow_ref=self._workflow_ref,
                        goal="Server image analysis via harness",
                    )
                )
            if result.status != RunResultStatus.COMPLETED:
                raise RuntimeError(f"Harness run {result.status.value}: {result.message}")

            detection_result = _workflow_output_to_detection_result(result.output)
            width, height = _read_image_size(Path(temp_path))
            if width > 0 and height > 0:
                detection_result.image_width = width
                detection_result.image_height = height
            logger.info(
                f"Harness analysis completed: run_id={result.run_id}",
                extra={"event": "harness_analysis_completed"},
            )
            return ServerAnalysisOutput(
                detection_result=detection_result,
                model_name=self.model_name,
                model_version=self.model_version,
                harness_run_id=result.run_id,
                harness_checkpoint_path=str(result.checkpoint_path)
                if result.checkpoint_path else None,
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class LegacyDetectorAnalysisRunner:
    """Adapter around existing detector implementations."""

    def __init__(self, detector):
        self.detector = detector

    @property
    def model_name(self) -> str:
        return self.detector.model_name

    @property
    def model_version(self) -> str:
        return self.detector.model_version

    def analyze_url(
        self,
        file_url: str,
        media_type: str = "Image",
    ) -> ServerAnalysisOutput:
        if _is_video(media_type, ""):
            raise ValueError("URL inference path currently supports image media only")
        if not hasattr(self.detector, "detect_image_url"):
            raise ValueError("Configured detector does not support URL image inference")

        detection_result = self.detector.detect_image_url(file_url)
        return ServerAnalysisOutput(
            detection_result=detection_result,
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def analyze_media(
        self,
        file_bytes: bytes,
        extension: str,
        media_type: str = "Image",
        request_id: str = "",
        artifact_dir: str = "",
        public_base_url: str = "",
        artifact_url_path: str = "/artifacts",
        jpeg_quality: int = 90,
    ) -> ServerAnalysisOutput:
        if _is_video(media_type, extension):
            from server_pc.app.video_processor import process_video

            detection_result = process_video(
                self.detector,
                file_bytes,
                extension,
                request_id=request_id,
                artifact_dir=artifact_dir,
                public_base_url=public_base_url,
                artifact_url_path=artifact_url_path,
                jpeg_quality=jpeg_quality,
            )
        elif hasattr(self.detector, "detect_image_bytes"):
            detection_result = self.detector.detect_image_bytes(file_bytes)
        else:
            import cv2
            import numpy as np

            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image bytes")
            detection_result = self.detector.detect_image(image)

        return ServerAnalysisOutput(
            detection_result=detection_result,
            model_name=self.model_name,
            model_version=self.model_version,
        )


def create_server_analysis_runner(settings, repo_root: Path):
    backend = settings.inference_backend.lower().strip()
    if backend == "harness":
        return HarnessAnalysisRunner(
            repo_root=repo_root,
            checkpoint_dir=Path(settings.harness_checkpoint_dir),
            workflow_ref=settings.harness_workflow_ref,
        )
    if backend == "roboflow":
        from server_pc.app.roboflow_detector import ServerRoboflowWorkflowDetector

        detector = ServerRoboflowWorkflowDetector(
            api_key=settings.roboflow_api_key,
            timeout_seconds=settings.roboflow_timeout,
            max_retries=settings.roboflow_max_retries,
            retry_base_delay=settings.roboflow_retry_base_delay,
            api_url=settings.roboflow_api_url,
            workspace_name=settings.roboflow_workspace_name,
            workflow_id=settings.roboflow_workflow_id,
            frame_sample_interval=settings.frame_sample_interval,
            max_frames_per_video=settings.max_frames_per_video,
            dedup_iou_threshold=settings.dedup_iou_threshold,
            max_detections_per_class=settings.max_detections_per_class,
        )
        return LegacyDetectorAnalysisRunner(detector)
    if backend == "local":
        from server_pc.app.detector import ServerRfDetrDetector

        detector = ServerRfDetrDetector(
            model_path=settings.rfdetr_model_path,
            config_path=settings.rfdetr_config_path,
            image_size=settings.rfdetr_image_size,
            conf_threshold=settings.rfdetr_conf_threshold,
            device=settings.rfdetr_device,
            frame_sample_interval=settings.frame_sample_interval,
            max_frames_per_video=settings.max_frames_per_video,
            dedup_iou_threshold=settings.dedup_iou_threshold,
            max_detections_per_class=settings.max_detections_per_class,
        )
        return LegacyDetectorAnalysisRunner(detector)
    raise ValueError("SERVER_INFERENCE_BACKEND must be 'harness', 'roboflow', or 'local'")


def _workflow_output_to_detection_result(output) -> DetectionResult:
    detections = []
    image_width = 0
    image_height = 0

    for item in output.detections:
        category = map_class_to_category(item.label)
        if category is None:
            continue
        detections.append(
            Detection(
                category_code=category,
                confidence=item.confidence,
                bounding_box=BoundingBox(
                    x=item.bounding_box.x,
                    y=item.bounding_box.y,
                    width=item.bounding_box.width,
                    height=item.bounding_box.height,
                ),
            )
        )

    return DetectionResult(
        detections=detections,
        image_width=image_width,
        image_height=image_height,
        frame_count=1,
    )


def _read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return 0, 0


def _is_video(media_type: str, extension: str) -> bool:
    return media_type.lower() == "video" or extension.lower() in _VIDEO_EXTENSIONS
