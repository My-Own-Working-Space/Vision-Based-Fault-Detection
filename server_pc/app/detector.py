"""
ServerRfDetrDetector — RF-DETR inference for PC/Server.

Implements the shared Detector protocol. Optimized for higher-accuracy
offline/asynchronous analysis with GPU acceleration when available.
"""

import os
import cv2
import numpy as np
from typing import List, Optional

from shared.schemas.analysis_result import (
    BoundingBox,
    Detection,
    DetectionResult,
)
from shared.services.class_mapping import map_class_to_category
from shared.utils.bbox import normalize_bbox
from shared.utils.logging import get_logger
from server_pc.app.metrics import observe_inference_duration

logger = get_logger("server_detector")


def _resolve_device(requested: str) -> str:
    """Resolve the inference device. 'auto' tries CUDA then CPU."""
    if requested.lower() == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA is available, using GPU")
                return "cuda"
        except ImportError:
            pass
        logger.info("CUDA not available, falling back to CPU")
        return "cpu"
    return requested


class ServerRfDetrDetector:
    """RF-DETR detector for high-accuracy server-side analysis.

    Implements the Detector protocol defined in
    shared.schemas.detector_interface.

    RF-DETR (Roboflow DETR) is a real-time detection transformer.
    This implementation uses the rfdetr Python package when available,
    with fallback to a generic Hugging Face transformers-based approach.
    """

    def __init__(
        self,
        model_path: str = "",
        config_path: str = "",
        image_size: int = 560,
        conf_threshold: float = 0.3,
        device: str = "auto",
        frame_sample_interval: int = 5,
        max_frames_per_video: int = 500,
        dedup_iou_threshold: float = 0.4,
        max_detections_per_class: int = 10,
    ):
        """Initialize the RF-DETR detector.

        Args:
            model_path: Path to RF-DETR weights file.
            config_path: Path to RF-DETR config (if needed).
            image_size: Input resolution for inference.
            conf_threshold: Confidence threshold for detections.
            device: 'auto', 'cuda', or 'cpu'.
            frame_sample_interval: Process every Nth frame for video.
            max_frames_per_video: Max frames to process per video.
            dedup_iou_threshold: IoU threshold for deduplication.
            max_detections_per_class: Max detections per class in video.
        """
        self._conf_threshold = conf_threshold
        self._image_size = image_size
        self._device = _resolve_device(device)
        self._frame_sample_interval = frame_sample_interval
        self._max_frames_per_video = max_frames_per_video
        self._dedup_iou_threshold = dedup_iou_threshold
        self._max_detections_per_class = max_detections_per_class

        self._model_name = "RF-DETR"
        self._model_version = "1.0.0"
        self._model = None
        self._class_names = {}

        logger.info(
            f"Loading RF-DETR model (device={self._device})",
            extra={"event": "model_load_start", "model": "RF-DETR"},
        )

        self._load_model(model_path, config_path)

        logger.info(
            "RF-DETR model loaded successfully",
            extra={"event": "model_load_complete", "model": "RF-DETR"},
        )

    def _load_model(self, model_path: str, config_path: str):
        """Load the RF-DETR model.

        Tries the rfdetr package first, then falls back to a
        generic transformers-based approach.
        """
        try:
            # Try rfdetr package (pip install rfdetr)
            from rfdetr import RFDETRBase, RFDETRLarge
            import torch
            import math

            # Default parameters
            num_classes = None
            resolution = self._image_size
            patch_size = 14
            state_dict = None

            # Load checkpoint first to infer parameters if file exists
            if model_path and os.path.exists(model_path):
                logger.info(f"Analyzing custom weights file to infer architecture: {model_path}")
                checkpoint = torch.load(model_path, map_location="cpu")
                state_dict = checkpoint.get("model", checkpoint)

                # Infer patch_size
                for k, v in state_dict.items():
                    if "patch_embeddings.projection.weight" in k:
                        patch_size = v.shape[2]
                        break

                # Infer num_classes
                for k, v in state_dict.items():
                    if "class_embed.weight" in k:
                        num_classes = v.shape[0] - 1
                        break

                # Infer resolution
                for k, v in state_dict.items():
                    if "position_embeddings" in k:
                        num_patches = v.shape[1] - 1
                        resolution = int(math.sqrt(num_patches) * patch_size)
                        break

                logger.info(
                    f"Inferred checkpoint parameters: patch_size={patch_size}, "
                    f"num_classes={num_classes}, resolution={resolution}"
                )

            # Build kwargs for the constructor
            constructor_kwargs = {
                "resolution": resolution,
                "patch_size": patch_size,
                "pretrain_weights": None, # Bypass default weights validation/download
            }
            if num_classes is not None:
                constructor_kwargs["num_classes"] = num_classes

            if "large" in model_path.lower():
                self._model = RFDETRLarge(**constructor_kwargs)
                self._model_name = "RF-DETR-Large"
            else:
                self._model = RFDETRBase(**constructor_kwargs)
                self._model_name = "RF-DETR-Base"

            # Load custom weights if state_dict is available
            if state_dict is not None:
                # Filter out keys with size mismatch to avoid PyTorch RuntimeError
                model_state = self._model.model.model.state_dict()
                filtered_state_dict = {}
                for k, v in state_dict.items():
                    if k in model_state:
                        if v.shape == model_state[k].shape:
                            filtered_state_dict[k] = v
                        else:
                            logger.warning(
                                f"Skipping key '{k}' due to size mismatch: "
                                f"checkpoint {list(v.shape)} vs model {list(model_state[k].shape)}"
                            )
                    else:
                        logger.warning(f"Skipping unexpected key '{k}' from checkpoint")

                self._model.model.model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"Successfully loaded custom weights from: {model_path}")

            logger.info("Using rfdetr package for inference")
            self._backend = "rfdetr"
            return

        except ImportError:
            logger.info("rfdetr package not found, trying transformers...")

        try:
            # Fallback: Hugging Face RT-DETR
            from transformers import (
                RTDetrForObjectDetection,
                RTDetrImageProcessor,
            )

            if model_path and os.path.isdir(model_path):
                self._model = RTDetrForObjectDetection.from_pretrained(model_path)
                self._processor = RTDetrImageProcessor.from_pretrained(model_path)
            else:
                self._model = RTDetrForObjectDetection.from_pretrained(
                    "PekingU/rtdetr_r50vd"
                )
                self._processor = RTDetrImageProcessor.from_pretrained(
                    "PekingU/rtdetr_r50vd"
                )

            import torch
            self._model.to(self._device)
            self._model.eval()

            # Build class name mapping from model config
            if hasattr(self._model.config, "id2label"):
                self._class_names = self._model.config.id2label

            self._model_name = "RT-DETR"
            self._backend = "transformers"
            logger.info("Using HuggingFace transformers for inference")
            return

        except ImportError:
            logger.warning("transformers package not found")

        # Final fallback: mock detector for testing
        logger.warning(
            "No RF-DETR backend available. Using mock detector. "
            "Install rfdetr or transformers to enable real inference.",
            extra={"event": "model_load_mock"},
        )
        self._backend = "mock"
        self._model_name = "RF-DETR-Mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def detect_image(self, image: np.ndarray) -> DetectionResult:
        """Run RF-DETR inference on a single image.

        Args:
            image: OpenCV BGR image as numpy array.

        Returns:
            DetectionResult with normalized bounding boxes.
        """
        h, w = image.shape[:2]
        detections: List[Detection] = []

        if self._backend == "rfdetr":
            detections = self._detect_rfdetr(image, w, h)
        elif self._backend == "transformers":
            detections = self._detect_transformers(image, w, h)
        else:
            # Mock — return empty detections
            pass

        logger.info(
            f"Image inference: {len(detections)} detections ({w}x{h})",
            extra={"event": "inference_image_complete"},
        )

        return DetectionResult(
            detections=detections,
            image_width=w,
            image_height=h,
            frame_count=1,
        )

    def detect_video(self, video_path: str) -> DetectionResult:
        """Run RF-DETR inference on a video file with denser sampling.

        Args:
            video_path: Path to the video file.

        Returns:
            DetectionResult with deduplicated detections.
        """
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

                    if self._backend == "rfdetr":
                        frame_dets = self._detect_rfdetr(frame, w, h)
                    elif self._backend == "transformers":
                        frame_dets = self._detect_transformers(frame, w, h)
                    else:
                        frame_dets = []

                    # Add timestamp and frame index
                    for det in frame_dets:
                        det.timestamp_ms = timestamp_ms
                        det.frame_index = frame_idx
                        all_detections.append(det)

                    processed_count += 1

                frame_idx += 1
        finally:
            cap.release()

        # Deduplicate
        deduped = self._deduplicate(all_detections)

        logger.info(
            f"Video inference: {len(deduped)} detections "
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

    def _detect_rfdetr(
        self, image: np.ndarray, w: int, h: int
    ) -> List[Detection]:
        """Run inference using the rfdetr package."""
        from PIL import Image as PILImage

        # Convert BGR to RGB PIL Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        # rfdetr returns a Detections object
        with observe_inference_duration("rfdetr"):
            results = self._model.predict(pil_img, threshold=self._conf_threshold)

        detections: List[Detection] = []

        # rfdetr uses supervision-style output
        if hasattr(results, "xyxy"):
            for i in range(len(results.xyxy)):
                x1, y1, x2, y2 = results.xyxy[i]
                conf = float(results.confidence[i])
                cls_id = int(results.class_id[i])

                if conf < self._conf_threshold:
                    continue

                # Get class name
                cls_name = str(cls_id)
                if hasattr(results, "data") and "class_name" in results.data:
                    cls_name = results.data["class_name"][i]
                elif hasattr(self._model, "model") and hasattr(
                    self._model.model, "names"
                ):
                    names = self._model.model.names
                    if isinstance(names, dict):
                        cls_name = names.get(cls_id, str(cls_id))
                    elif isinstance(names, list) and cls_id < len(names):
                        cls_name = names[cls_id]

                category = map_class_to_category(cls_name)
                if category is None:
                    continue

                bbox = normalize_bbox(
                    float(x1), float(y1), float(x2), float(y2), w, h
                )
                detections.append(
                    Detection(
                        category_code=category,
                        confidence=conf,
                        bounding_box=BoundingBox(**bbox),
                    )
                )

        return detections

    def _detect_transformers(
        self, image: np.ndarray, w: int, h: int
    ) -> List[Detection]:
        """Run inference using HuggingFace transformers RT-DETR."""
        import torch
        from PIL import Image as PILImage

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            with observe_inference_duration("transformers"):
                outputs = self._model(**inputs)

        target_sizes = torch.tensor([[h, w]]).to(self._device)
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self._conf_threshold,
        )[0]

        detections: List[Detection] = []

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            conf = float(score)
            if conf < self._conf_threshold:
                continue

            cls_id = int(label)
            cls_name = self._class_names.get(cls_id, str(cls_id))

            category = map_class_to_category(cls_name)
            if category is None:
                continue

            x1, y1, x2, y2 = map(float, box)
            bbox = normalize_bbox(x1, y1, x2, y2, w, h)
            detections.append(
                Detection(
                    category_code=category,
                    confidence=conf,
                    bounding_box=BoundingBox(**bbox),
                )
            )

        return detections

    def _deduplicate(
        self, detections: List[Detection]
    ) -> List[Detection]:
        """Deduplicate detections by category and spatial overlap."""
        from shared.utils.bbox import calculate_iou

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
                is_dup = False
                for existing in kept:
                    existing_box = existing.bounding_box.model_dump()
                    if (
                        calculate_iou(box, existing_box)
                        > self._dedup_iou_threshold
                    ):
                        is_dup = True
                        break

                if not is_dup:
                    kept.append(det)

            result.extend(kept)

        return result
