from typing import Any
import os
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

from messaging import Message, MessageStatus
from agent.vision_agent import VisionAgent


class YOLOVisionAgent(VisionAgent):
    """Concrete implementation of VisionAgent using a YOLO object detection model."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        super().__init__()  # Passes name="Vision" up to BaseAgent
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Initializes and loads the YOLO model into memory."""
        print(f"[YOLOVisionAgent] Initializing model from '{self.model_path}'...")
        self.model = YOLO(self.model_path)

    def _detect_objects(self, message: Message, payload: dict[str, Any]) -> Message:
        """Runs object detection on the provided image."""
        image_path = payload.get("image_path")

        if not image_path:
            return self.create_response(
                request=message,
                payload={
                    "image_path": None,
                    "objects": [],
                    "count": 0,
                    "detections": [],
                    "warning": "No 'image_path' provided in payload.",
                },
            )

        if not os.path.isfile(image_path):
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Image file not found: '{image_path}'",
            )

        frame = cv2.imread(image_path)
        if frame is None:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Failed to read image (unsupported format or corrupt file): '{image_path}'",
            )

        print(f"[YOLOVisionAgent] Detecting objects in '{image_path}'...")
        confidence = payload.get("confidence", 0.25)

        results = self.model(frame, conf=confidence, verbose=False)
        detections = self._build_detections(results)
        object_names = [d["class_name"] for d in detections]

        return self.create_response(
            request=message,
            payload={
                "image_path": image_path,
                "objects": object_names,
                "count": len(object_names),
                "detections": detections,
            },
        )

    def _analyze_scene(self, message: Message, payload: dict[str, Any]) -> Message:
        """Analyzes the overall context or summary of the scene."""
        image_path = payload.get("image_path")

        if not image_path:
            return self.create_response(
                request=message,
                payload={
                    "image_path": None,
                    "summary": "No image provided for scene analysis.",
                    "detections": [],
                    "warning": "No 'image_path' provided in payload.",
                },
            )

        if not os.path.isfile(image_path):
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Image file not found: '{image_path}'",
            )

        frame = cv2.imread(image_path)
        if frame is None:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Failed to read image (unsupported format or corrupt file): '{image_path}'",
            )

        print(f"[YOLOVisionAgent] Analyzing scene for '{image_path}'...")
        confidence = payload.get("confidence", 0.25)

        results = self.model(frame, conf=confidence, verbose=False)
        detections = self._build_detections(results)
        summary = self._summarize_scene(detections)

        return self.create_response(
            request=message,
            payload={
                "image_path": image_path,
                "summary": summary,
                "detections": detections,
            },
        )


    def _build_detections(self, results) -> list[dict[str, Any]]:
        detections = []

        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {}) or {}

            if not boxes:
                continue

            for box in boxes:
                coords = self._extract_box_coordinates(box)
                if coords is None:
                    continue

                cls_value = self._first_value(getattr(box, "cls", None))
                confidence_value = self._first_value(getattr(box, "conf", None))
                cls_id = int(cls_value) if cls_value is not None else -1
                confidence = float(confidence_value) if confidence_value is not None else 0.0
                class_name = names.get(cls_id, str(cls_id))

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": coords,
                    }
                )

        return detections

    def _extract_box_coordinates(self, box):
        xyxy = getattr(box, "xyxy", None)
        if xyxy is None:
            return None

        if hasattr(xyxy, "tolist"):
            xyxy = xyxy.tolist()

        while isinstance(xyxy, (list, tuple)) and len(xyxy) == 1:
            xyxy = xyxy[0]

        if hasattr(xyxy, "tolist"):
            xyxy = xyxy.tolist()

        if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
            return None

        try:
            return tuple(float(value) for value in xyxy[:4])
        except (TypeError, ValueError):
            return None

    def _first_value(self, value):
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    def _summarize_scene(self, detections: list[dict[str, Any]]) -> str:
        """Builds a human-readable scene summary from detections (counts by class,
        sorted by frequency, highest-confidence class highlighted)."""
        if not detections:
            return "No recognizable objects detected in the scene."

        counts: dict[str, int] = {}
        for d in detections:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1

        sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        parts = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in sorted_items]

        if len(parts) == 1:
            objects_str = parts[0]
        else:
            objects_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        return f"Scene contains {objects_str}."