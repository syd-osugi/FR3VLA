"""
DINOv2 Grounding / Segmentation Interface
-----------------------------------------
Turns a camera image plus a requested object name into segmentation detections.

Important model note:
    Plain DINOv2 is a visual feature backbone. It does not understand arbitrary
    text prompts by itself. This adapter expects a DINOv2-family semantic
    segmentation checkpoint with an id2label map. The requested object text is
    matched against those labels, then the matching label mask is returned.

Typical usage:
    detector = DINOv2SegmentationDetector()
    detections = detector.detect(image_bgr, classes_to_find="cup")
    by_camera = detector.detect_in_images({"d435": d435_bgr, "d405": d405_bgr}, "cup")

Returned detections are dictionaries with:
    label, score, pixel, bbox_xywh, bbox_xyxy, area_px, mask

The class is lazy-loaded so importing this module never downloads or initializes
torch/transformers. Model loading happens on first detect() or explicit load().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

import config as cfg
from vision.base_classes import BaseDetector


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "find",
    "for",
    "get",
    "grab",
    "in",
    "locate",
    "of",
    "on",
    "pick",
    "please",
    "the",
    "this",
    "to",
    "up",
}

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}

BACKGROUND_LABELS = {
    "background",
    "ignore",
    "ignored",
    "other",
    "unlabeled",
    "unknown",
    "void",
}


def normalize_label(value: Any) -> str:
    """Normalize labels/prompts for forgiving text matching."""
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def request_terms(classes_to_find) -> list[str]:
    """
    Convert a requested object string/list into searchable terms.

    "red coffee mug" becomes ["red coffee mug", "coffee", "mug"]. Color words
    are intentionally dropped from token terms because semantic segmentation
    heads usually label object categories, not colors.
    """
    if classes_to_find is None:
        return []
    if isinstance(classes_to_find, str):
        raw_items = [classes_to_find]
    else:
        try:
            raw_items = list(classes_to_find)
        except TypeError:
            raw_items = [classes_to_find]

    terms = []
    for item in raw_items:
        phrase = normalize_label(item)
        if not phrase:
            continue
        terms.append(phrase)
        for token in phrase.split():
            if token in DEFAULT_STOPWORDS or token in COLOR_WORDS:
                continue
            if len(token) >= 3:
                terms.append(token)

    deduped = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return deduped


def parse_label_aliases(value) -> dict[str, list[str]]:
    """
    Parse optional alias mappings.

    Expected JSON shape:
        {"mug": ["cup", "coffee cup"], "block": ["cube"]}
    """
    if not value:
        return {}
    if isinstance(value, dict):
        raw = value
    else:
        raw = json.loads(str(value))

    aliases = {}
    for key, values in raw.items():
        key_norm = normalize_label(key)
        if isinstance(values, str):
            values = [values]
        aliases[key_norm] = [normalize_label(item) for item in values if normalize_label(item)]
    return aliases


def expand_terms_with_aliases(terms: list[str], aliases: dict[str, list[str]]) -> list[str]:
    expanded = list(terms)
    for term in terms:
        for alias_key, alias_values in aliases.items():
            if term == alias_key or term in alias_values:
                expanded.extend([alias_key, *alias_values])

    deduped = []
    for term in expanded:
        if term and term not in deduped:
            deduped.append(term)
    return deduped


def label_matches_terms(label: str, terms: list[str]) -> bool:
    """Return true when a model label looks like the requested object."""
    label_norm = normalize_label(label)
    if not terms:
        return True
    label_tokens = set(label_norm.split())
    for term in terms:
        term_tokens = set(term.split())
        if term == label_norm:
            return True
        if term in label_norm or label_norm in term:
            return True
        if term_tokens and term_tokens.issubset(label_tokens):
            return True
    return False


def bbox_xywh_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x_min = int(xs.min())
    y_min = int(ys.min())
    return [
        x_min,
        y_min,
        int(xs.max() - x_min + 1),
        int(ys.max() - y_min + 1),
    ]


def component_detections_from_mask(
    mask: np.ndarray,
    confidence_map: np.ndarray,
    *,
    label: str,
    label_id: int,
    requested_class: str | None,
    min_area_px: int,
    model_name: str,
) -> list[dict]:
    """Split a class mask into connected components and detection records."""
    mask = np.asarray(mask).astype(bool)
    confidence_map = np.asarray(confidence_map, dtype=float)
    if mask.shape != confidence_map.shape:
        raise ValueError("mask and confidence_map must have matching shapes")

    try:
        import cv2
    except ModuleNotFoundError:
        cv2 = None

    detections = []
    if cv2 is not None:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            8,
        )
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue
            component_mask = labels == component_id
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[component_id]
            detections.append(
                {
                    "label": label,
                    "label_id": int(label_id),
                    "requested_class": requested_class,
                    "score": float(confidence_map[component_mask].mean()),
                    "confidence": float(confidence_map[component_mask].mean()),
                    "pixel": [int(round(cx)), int(round(cy))],
                    "center_pixel": [int(round(cx)), int(round(cy))],
                    "bbox_xywh": [x, y, w, h],
                    "bbox_xyxy": [x, y, x + w - 1, y + h - 1],
                    "area_px": area,
                    "mask": component_mask.astype(np.uint8),
                    "model": model_name,
                    "source": "dinov2_semantic_segmentation",
                }
            )
        return detections

    bbox = bbox_xywh_from_mask(mask)
    if bbox is None:
        return []
    area = int(mask.sum())
    if area < min_area_px:
        return []
    x, y, w, h = bbox
    ys, xs = np.nonzero(mask)
    center = [int(round(float(xs.mean()))), int(round(float(ys.mean())))]
    return [
        {
            "label": label,
            "label_id": int(label_id),
            "requested_class": requested_class,
            "score": float(confidence_map[mask].mean()),
            "confidence": float(confidence_map[mask].mean()),
            "pixel": center,
            "center_pixel": center,
            "bbox_xywh": bbox,
            "bbox_xyxy": [x, y, x + w - 1, y + h - 1],
            "area_px": area,
            "mask": mask.astype(np.uint8),
            "model": model_name,
            "source": "dinov2_semantic_segmentation",
            "reason": "OpenCV unavailable; component split was not applied.",
        }
    ]


@dataclass
class DINOv2Runtime:
    torch: Any
    image_processor: Any
    model: Any
    image_class: Any
    device: str


class DINOv2SegmentationDetector(BaseDetector):
    """
    DINOv2-family semantic segmentation detector.

    Configure with env vars:
        DINOV2_MODEL_NAME
        DINOV2_DEVICE
        DINOV2_SCORE_THRESHOLD
        DINOV2_MIN_AREA_PX
        DINOV2_LABEL_ALIASES_JSON
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        score_threshold: float | None = None,
        min_area_px: int | None = None,
        max_detections: int | None = None,
        label_aliases: dict[str, list[str]] | str | None = None,
        image_color_order: str = "BGR",
    ):
        self.model_name = model_name or cfg.DINOV2_MODEL_NAME
        self.device = device or cfg.DINOV2_DEVICE
        self.score_threshold = (
            float(score_threshold)
            if score_threshold is not None
            else float(cfg.DINOV2_SCORE_THRESHOLD)
        )
        self.min_area_px = int(min_area_px if min_area_px is not None else cfg.DINOV2_MIN_AREA_PX)
        self.max_detections = int(
            max_detections
            if max_detections is not None
            else cfg.DINOV2_MAX_DETECTIONS
        )
        self.label_aliases = parse_label_aliases(
            label_aliases
            if label_aliases is not None
            else cfg.DINOV2_LABEL_ALIASES_JSON
        )
        self.image_color_order = image_color_order.upper()
        self._runtime: DINOv2Runtime | None = None
        self.last_result: dict | None = None

    def load(self):
        """Load torch/transformers and initialize the configured checkpoint."""
        if self._runtime is not None:
            return self

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("DINOv2 detector requires torch.") from exc

        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("DINOv2 detector requires Pillow.") from exc

        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except ModuleNotFoundError as exc:
            raise RuntimeError("DINOv2 detector requires transformers.") from exc

        resolved_device = self._resolve_device(torch)
        image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        try:
            model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                "The configured DINOv2 checkpoint could not be loaded as a semantic "
                "segmentation model. Use a DINOv2-family checkpoint with a segmentation "
                "head and id2label labels, not a bare facebook/dinov2-* encoder."
            ) from exc

        model.to(resolved_device)
        model.eval()
        self._runtime = DINOv2Runtime(
            torch=torch,
            image_processor=image_processor,
            model=model,
            image_class=Image,
            device=resolved_device,
        )
        return self

    def detect(self, image, classes_to_find=None):
        """
        Segment the requested object in a BGR camera image.

        Args:
            image: OpenCV/RealSense image, normally BGR uint8 HxWx3.
            classes_to_find: requested object text or list of object names.

        Returns:
            list[dict]: detections sorted by confidence and area.
        """
        self.load()
        assert self._runtime is not None

        pil_image = self._to_pil_rgb(image)
        width, height = pil_image.size
        requested_terms = expand_terms_with_aliases(
            request_terms(classes_to_find),
            self.label_aliases,
        )
        target_labels = self._target_label_ids(requested_terms)

        if not target_labels:
            self.last_result = {
                "valid": False,
                "reason": (
                    f"No model label matched request {classes_to_find!r}. "
                    "Check the checkpoint id2label map or DINOV2_LABEL_ALIASES_JSON."
                ),
                "requested_terms": requested_terms,
                "available_labels": self.available_labels(),
            }
            return []

        torch = self._runtime.torch
        inputs = self._runtime.image_processor(images=pil_image, return_tensors="pt")
        inputs = {
            key: value.to(self._runtime.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self._runtime.model(**inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("Configured DINOv2 segmentation model did not return logits.")
            logits = torch.nn.functional.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            predicted_ids = probabilities.argmax(axis=0)

        detections = []
        for label_id, label, requested_class in target_labels:
            class_probability = probabilities[label_id]
            mask = (predicted_ids == label_id) & (class_probability >= self.score_threshold)
            detections.extend(
                component_detections_from_mask(
                    mask,
                    class_probability,
                    label=label,
                    label_id=label_id,
                    requested_class=requested_class,
                    min_area_px=self.min_area_px,
                    model_name=self.model_name,
                )
            )

        detections.sort(key=lambda item: (item["score"], item["area_px"]), reverse=True)
        detections = detections[: self.max_detections]
        self.last_result = {
            "valid": bool(detections),
            "requested_terms": requested_terms,
            "matched_labels": [
                {"label_id": label_id, "label": label, "requested_class": requested_class}
                for label_id, label, requested_class in target_labels
            ],
            "detection_count": len(detections),
            "image_size": [width, height],
        }
        if not detections:
            self.last_result["reason"] = (
                "Matched labels were present, but no component passed the score/area thresholds."
            )
        return detections

    def detect_in_images(self, images: dict[str, Any], classes_to_find=None) -> dict[str, list[dict]]:
        """
        Segment the requested object in multiple camera images.

        Args:
            images: mapping like {"d435": d435_bgr, "d405": d405_bgr}
            classes_to_find: requested object text or list of object names.

        Returns:
            dict mapping camera name to detection list. Each detection includes
            a "camera" field in addition to the normal detect() fields.
        """
        results = {}
        for camera_name, image in images.items():
            detections = self.detect(image, classes_to_find=classes_to_find)
            camera_detections = []
            for detection in detections:
                annotated = dict(detection)
                annotated["camera"] = camera_name
                camera_detections.append(annotated)
            results[camera_name] = camera_detections
        return results

    def available_labels(self) -> dict[int, str]:
        """Return the loaded model's id2label mapping, if available."""
        if self._runtime is None:
            return {}
        id2label = getattr(self._runtime.model.config, "id2label", {}) or {}
        return {int(key): str(value) for key, value in id2label.items()}

    def _target_label_ids(self, requested_terms: list[str]) -> list[tuple[int, str, str | None]]:
        labels = self.available_labels()
        if not labels:
            raise RuntimeError(
                "Configured segmentation checkpoint has no id2label map; cannot match requested objects."
            )

        matches = []
        for label_id, label in labels.items():
            label_norm = normalize_label(label)
            if not requested_terms and label_norm in BACKGROUND_LABELS:
                continue
            if label_matches_terms(label, requested_terms):
                matched_term = None
                for term in requested_terms:
                    if label_matches_terms(label, [term]):
                        matched_term = term
                        break
                matches.append((int(label_id), str(label), matched_term))
        return matches

    def _resolve_device(self, torch_module) -> str:
        configured = str(self.device or "auto").strip().lower()
        if configured == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return configured

    def _to_pil_rgb(self, image):
        if hasattr(image, "convert"):
            return image.convert("RGB")

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        if self.image_color_order == "BGR":
            array = array[:, :, ::-1]
        elif self.image_color_order != "RGB":
            raise ValueError("image_color_order must be BGR or RGB")

        assert self._runtime is not None
        return self._runtime.image_class.fromarray(array, mode="RGB")


# Backward-compatible names for callers that use different capitalization.
DINO2VSegmentationDetector = DINOv2SegmentationDetector
Dinov2SegmentationDetector = DINOv2SegmentationDetector


class LLMVisualDetector(DINOv2SegmentationDetector):
    """
    Backward-compatible detector name.

    Older code imported LLMVisualDetector from this module when vision grounding
    was a placeholder. It now points at the DINOv2 segmentation implementation.
    """
